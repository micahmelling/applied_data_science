import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer

from ds_helpers import aws, db
from helpers.model_helpers import create_train_test_split, create_x_y_split, FeaturesToDict, fill_missing_values
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
        try:
            df[col] = df[col].astype(float)
            df[col] = df[col].str.strip()
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


def create_training_data(original_training_df, production_df):
    original_training_df['target'] = 0
    production_df['target'] = 1
    training_df = pd.concat([original_training_df, production_df], axis=0)
    training_df.reset_index(inplace=True, drop=True)
    training_df.drop('acquired_date', 1, inplace=True)
    training_df = training_df.fillna(value=np.nan)
    x, y = create_x_y_split(training_df, 'target')
    x_train, x_test, y_train, y_test = create_train_test_split(x, y)
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    param_grid = {
        'model__max_depth': [5, 10, 15],
        'model__min_samples_leaf': [None, 3],
        'model__max_features': ['log2', 'sqrt']
    }

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', FunctionTransformer(fill_missing_values, validate=False,
                                        kw_args={'fill_value': 'unknown'})),
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer(sparse=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_transformer', numeric_transformer, selector(dtype_include='number')),
            ('categorical_transformer', categorical_transformer, selector(dtype_exclude='number'))
        ]
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier())
    ])

    search = GridSearchCV(pipeline, param_grid=param_grid, scoring='roc_auc', n_jobs=-1, cv=3)
    search.fit(x_train, y_train)
    best_pipeline = search.best_estimator_
    return best_pipeline


def evaluate_model(estimator, x_test, y_test):
    predictions = estimator.predict_proba(x_test)
    roc_auc = roc_auc_score(y_test, predictions[:, 1])
    return roc_auc


def determine_if_concept_shift_has_occurred(score, threshold):
    if score >= threshold:
        return True
    else:
        return False


def main(model_path, db_secret_name, scoring_threshold):
    db_conn = db.connect_to_mysql(aws.get_secrets_manager_secret(db_secret_name),
                                  ssl_path=os.path.join('data', 'rds-ca-2019-root.pem'))
    model_uid = extract_model_uid_from_path(model_path)
    query_start_time = get_query_start_timestamp(model_uid, db_conn)
    production_df = extract_production_data(query_start_time, MODEL_1_PATH, db_conn)
    original_training_df = recreate_data_used_for_training(model_uid)
    x_train, x_test, y_train, y_test = create_training_data(original_training_df, production_df)
    pipeline = train_model(x_train, y_train)
    model_score = evaluate_model(pipeline, x_test, y_test)
    shift_occurred = determine_if_concept_shift_has_occurred(roc_auc, scoring_threshold)
    insert_statement = f'''
        INSERT INTO churn_model.concept_shift (shift_occurred, metric_used, scoring_threshold, model_score, model_uid)
        VALUES ({shift_occurred}, 'roc_auc', {scoring_threshold}, {model_score}, '{model_uid}');
        '''
    db_conn.execute(insert_statement)
    print(f'Has concept shift occurred: {shift_occurred}')


if __name__ == "__main__":
    main(model_path=MODEL_1_PATH, db_secret_name='churn-model-mysql', scoring_threshold=0.55)
