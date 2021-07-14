import pandas as pd
import numpy as np
import os
import joblib

from scipy.stats import ks_2samp, chisquare

from ds_helpers import aws, db
from app_settings import MODEL_1_PATH, MODEL_FEATURES


def extract_model_uid_from_path(model_path):
    return model_path.split('/')[1]


def get_query_start_timestamp(model_uid, db_conn):
    query = f'''
    select training_timestamp
    from churn_model.model_meta_data
    where model_uid = '{model_uid}';
    '''
    df = pd.read_sql(query, db_conn)
    start_timestamp = df['training_timestamp'][0]
    return start_timestamp


def extract_production_data(start_timestamp, model_uid, db_conn):
    query = f'''
    select JSON_EXTRACT(input_output_payloads, "$.input.*") as "values",
    JSON_KEYS(input_output_payloads, "$.input") as "keys"
    from (
    select * from churn_model.model_logs
    where JSON_EXTRACT(input_output_payloads, "$.output.model_used") = 'model_1'
    and JSON_EXTRACT(input_output_payloads, "$.output.model_1_path") = '{model_uid}'

    union

    select * from churn_model.model_logs
    where JSON_EXTRACT(input_output_payloads, "$.output.model_used") = 'model_2'
    and JSON_EXTRACT(input_output_payloads, "$.output.model_2_path") = '{model_uid}'

    ) model_output
    where logging_timestamp >= '{start_timestamp}';'''
    df = pd.read_sql(query, db_conn)
    columns = df['keys'][0]
    columns = columns.strip('][').split(', ')
    columns = [c.replace('"', '') for c in columns]
    df.drop('keys', 1, inplace=True)
    df['values'] = df['values'].str.replace('[', '').str.replace(']', '')
    df = df['values'].str.split(',', expand=True)
    df.columns = columns
    df.drop(['uid', 'url', 'endpoint'], 1, inplace=True)
    for col in list(df):
        df[col] = df[col].str.replace('"', '')
        df[col] = df[col].str.strip()
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
    return df


def recreate_data_used_for_training(model_uid):
    path = os.path.join(model_uid, 'data')
    aws.download_folder_from_s3('churn-model-data-science-modeling', path)
    x_train = joblib.load(os.path.join(path, 'x_train.pkl'))
    x_train.reset_index(inplace=True, drop=True)
    x_test = joblib.load(os.path.join(path, 'x_test.pkl'))
    x_test.reset_index(inplace=True, drop=True)
    x_df = pd.concat([x_train, x_test], axis=0)
    x_df = x_df[MODEL_FEATURES]
    return x_df


def calculate_ks_statistic(training_df, production_df, feature):
    try:
        ks_result = ks_2samp(training_df[feature], production_df[feature])
        return ks_result[1]
    except KeyError:
        return 0.00


def prep_category_for_chi_squared(training_df, production_df, feature):
    try:
        training_feature_grouped = pd.DataFrame(training_df.groupby(feature)[feature].count())
        training_feature_grouped.columns = ['train_count']
        training_feature_grouped['train_count'] = (training_feature_grouped['train_count'] /
                                                   training_feature_grouped['train_count'].sum()) * 1_000
        training_feature_grouped['train_count'] = training_feature_grouped['train_count'] + 1
        training_feature_grouped.reset_index(inplace=True)
    except KeyError:
        training_feature_grouped = pd.DataFrame({'train_count': [], feature: []})

    try:
        production_feature_grouped = pd.DataFrame(production_df.groupby(feature)[feature].count())
        production_feature_grouped.columns = ['prod_count']
        production_feature_grouped['prod_count'] = (production_feature_grouped['prod_count'] /
                                                    production_feature_grouped['prod_count'].sum()) * 1_000
        production_feature_grouped['prod_count'] = production_feature_grouped['prod_count'] + 1
        production_feature_grouped.reset_index(inplace=True)
    except KeyError:
        production_feature_grouped = pd.DataFrame({'prod_count': [], feature: []})

    merged_df = pd.merge(training_feature_grouped, production_feature_grouped, how='outer', on=feature)
    merged_df.fillna(0, inplace=True)
    merged_df['train_count'] = merged_df['train_count'].astype(int)
    merged_df['prod_count'] = merged_df['prod_count'].astype(int)
    return merged_df


def calculate_chi_squared_statistic(training_series, production_series):
    chi_result = chisquare(f_obs=production_series, f_exp=training_series)
    return chi_result[1]


def main(model_path, db_secret_name, p_value_cutoff):
    db_conn = db.connect_to_mysql(aws.get_secrets_manager_secret(db_secret_name),
                                  ssl_path=os.path.join('data', 'rds-ca-2019-root.pem'))
    model_uid = extract_model_uid_from_path(model_path)
    query_start_time = get_query_start_timestamp(model_uid, db_conn)
    production_df = extract_production_data(query_start_time, MODEL_1_PATH, db_conn)
    original_training_df = recreate_data_used_for_training(model_uid)

    cat_production_df = production_df.select_dtypes(include='object')
    num_production_df = production_df.select_dtypes(exclude='object')
    cat_training_df = original_training_df.select_dtypes(include='object')
    num_training_df = original_training_df.select_dtypes(exclude='object')

    cat_columns = set(list(cat_production_df) + list(cat_training_df))
    num_columns = set(list(num_production_df) + list(num_training_df))
    main_drift_df = pd.DataFrame()

    for cat_col in cat_columns:
        temp_chi_squared_df = prep_category_for_chi_squared(cat_training_df, cat_production_df, cat_col)
        p_value = calculate_chi_squared_statistic(temp_chi_squared_df['train_count'],
                                                  temp_chi_squared_df['prod_count'])
        temp_drift_df = pd.DataFrame({'feature': [cat_col], 'p_value': [p_value]})
        main_drift_df = main_drift_df.append(temp_drift_df)

    for num_col in num_columns:
        p_value = calculate_ks_statistic(num_training_df, num_production_df, num_col)
        temp_drift_df = pd.DataFrame({'feature': [num_col], 'p_value': [p_value]})
        main_drift_df = main_drift_df.append(temp_drift_df)

    main_drift_df['shift_occurred'] = np.where(main_drift_df['p_value'] <= p_value_cutoff, True, False)
    main_drift_df['p_value_cutoff'] = p_value_cutoff
    db.write_dataframe_to_database(main_drift_df, 'churn_model', 'data_shift', db_conn)


if __name__ == "__main__":
    main(model_path=MODEL_1_PATH, db_secret_name='churn-model-mysql', p_value_cutoff=0.05)
