import pandas as pd
import joblib
import os

from ds_helpers import aws


def extract_model_uid_from_path(model_path):
    """
    Extracts the UID from model_path.

    :param model_path: path to the model
    :returns: str
    """
    return model_path.split('/')[1]


def get_query_start_timestamp(model_uid, db_conn):
    """
    Finds the timestamp when the model was trained.

    :param model_uid: model uid
    :param db_conn: SQLAlchemy connection
    """
    query = f'''
    select training_timestamp
    from churn_model.model_meta_data
    where model_uid = '{model_uid}';
    '''
    df = pd.read_sql(query, db_conn)
    start_timestamp = df['training_timestamp'][0]
    return start_timestamp


def extract_production_data(start_timestamp, model_uid, db_conn):
    """
    Retrieves input and output payloads for all requests associated with model_uid.

    :param start_timestamp: time to start the query
    :param model_uid: model uid
    :param db_conn: SQLAlchemy connection
    :returns: pandas dataframe
    """
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


def recreate_data_used_for_training(model_uid, model_features):
    """
    Pull data used for training the model with the specified model_uid.

    :param model_uid: model uid
    :returns: pandas dataframe
    """
    path = os.path.join(model_uid, 'data')
    aws.download_directory_from_s3('churn-model-data-science-modeling', path)
    x_train = joblib.load(os.path.join(path, 'x_train.pkl'))
    x_train.reset_index(inplace=True, drop=True)
    x_test = joblib.load(os.path.join(path, 'x_test.pkl'))
    x_test.reset_index(inplace=True, drop=True)
    x_df = pd.concat([x_train, x_test], axis=0)
    x_df = x_df[model_features]
    return x_df
