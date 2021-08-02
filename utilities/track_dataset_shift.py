import pandas as pd
import numpy as np

from scipy.stats import ks_2samp, chisquare

from ds_helpers import db

from data.db import make_mysql_connection
from modeling.config import FEATURE_DTYPE_MAPPING
from app_settings import MODEL_PATH
from helpers.utility_helpers import extract_model_uid_from_path, get_query_start_timestamp, extract_production_data, \
    recreate_data_used_for_training


def calculate_ks_statistic(training_series, production_series):
    """
    Calculates a KS Statistic between production and training data for two comparable series.

    :param training_series: series of training data
    :param production_series: series of production data
    :returns: float
    """
    try:
        ks_result = ks_2samp(training_series, production_series)
        return ks_result[1]
    except KeyError:
        return 0.00


def prep_category_for_chi_squared(training_df, production_df, feature):
    """
    Prepares a categorical feature for calculating a chi-squared statistic. Basically, it normalizes the sample sizes
    to be comparable.

    :param training_df: dataframe of training data
    :param production_df: dataframe of production data
    :param feature: categorical feature
    :returns: pandas dataframe
    """
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
    """
    Calculates a chi-squared statistic and p-value.

    :param training_series: series of training data
    :param production_series: series of production data
    :returns: p-value
    """
    chi_result = chisquare(f_obs=production_series, f_exp=training_series)
    return chi_result[1]


def main(model_path, db_secret_name, p_value_cutoff, model_features):
    """
    Determines if concept shift has occurred.

    :param model_path: path to the model
    :param db_secret_name: Secrets Manager secret with DB credentials
    :param p_value_cutoff: p-value for chi-squared calculation
    :param model_features: features used for modeling
    """
    db_conn = make_mysql_connection(db_secret_name)
    model_uid = extract_model_uid_from_path(model_path)
    query_start_time = get_query_start_timestamp(model_uid, db_conn)
    production_df = extract_production_data(query_start_time, model_uid, db_conn)
    original_training_df = recreate_data_used_for_training(model_uid, model_features)

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
        p_value = calculate_ks_statistic(num_training_df[num_col], num_production_df[num_col])
        temp_drift_df = pd.DataFrame({'feature': [num_col], 'p_value': [p_value]})
        main_drift_df = main_drift_df.append(temp_drift_df)

    main_drift_df['shift_occurred'] = np.where(main_drift_df['p_value'] <= p_value_cutoff, True, False)
    main_drift_df['p_value_cutoff'] = p_value_cutoff
    db.write_dataframe_to_database(main_drift_df, 'churn_model', 'data_shift', db_conn)


if __name__ == "__main__":
    main(
        model_path=MODEL_PATH,
        db_secret_name='churn-model-mysql',
        p_value_cutoff=0.05,
        model_features=list(FEATURE_DTYPE_MAPPING.keys())
    )
