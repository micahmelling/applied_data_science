import pandas as pd
import pathlib
import os
import sqlalchemy

from ds_helpers import db, aws
from cachetools import cached, TTLCache

from helpers.app_helpers import get_current_timestamp


SSL_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'rds-ca-2019-root.pem')


def make_mysql_connection(db_secret_name):
    """
    Establishes a connection with MySQL

    :param db_secret_name: Secrets Manager secret name with DB creds
    """
    return db.connect_to_mysql(aws.get_secrets_manager_secret(db_secret_name), ssl_path=SSL_PATH)


def get_training_data(db_conn):
    """
    Retrieves the training data.
    """
    return pd.read_sql('''select * from churn_model.churn_data;''', db_conn)


def log_model_metadata(model_uid, schema, db_conn):
    """
    Logs model metadata to a database table.

    :param model_uid: model uid
    :param schema: schema name
    :param db_conn: database connection
    """
    df = pd.DataFrame({
        'training_timestamp': [get_current_timestamp()],
        'model_uid': [model_uid]
    })
    df.to_sql(name='model_metadata', schema=schema, con=db_conn, if_exists='append', index=False)


def retrieve_app_config(schema, db_conn, environment):
    """
    Retrieves the application configuration from a database table.

    :param schema: name of the MySQL schema
    :param db_conn: database connection
    :param environment: ENVIRONMENT env variabele
    :return: dictionary with config keys and values
    """
    if environment in ['STAGE', 'LOCAL']:
        table_name = 'stage_config'
    else:
        table_name = 'prod_config'
    query = f'''
    select config_key, config_value 
    from {schema}.{table_name}
    where meta__inserted_at = (select max(meta__inserted_at) from {schema}.{table_name})
    ;'''
    df = pd.read_sql(query, db_conn)
    df = df.set_index('config_key')
    df_dict = df.to_dict().get('config_value')
    return df_dict


@cached(cache=TTLCache(maxsize=1, ttl=86_400))
def get_client_ltv_table(db_conn):
    """
    Gets the LTV for every client_id.

    :param db_conn: database connection
    :returns: pandas dataframe
    """
    query = '''
    select client_id, ltv
    from churn_model.client_ltv;
    '''
    df = pd.read_sql(query, db_conn)
    return df


def get_hashed_password_for_username(username, db_conn):
    """
    Gets the hashed_password for username.

    :param username: a username
    :param db_conn: database connection
    :returns: hashed password as string
    """
    query = f'''
    select password
    from churn_model.app_credentials
    where username = '{username}';
    '''
    df = pd.read_sql(query, db_conn)
    return df['password'][0]


def log_payloads_to_mysql(input_payload, output_payload, table_name, schema_name, db_secret_name):
    """
    Logs input and output payloads payloads to MySQL:

    :param input_payload: input payload
    :param output_payload: output_payload
    :param table_name: name of MySQL table
    :param schema_name: name of MySQL schema
    :param db_secret_name: Secrets Manager secret with DB creds
    """
    uid = input_payload.get('uid')
    logging_timestamp = output_payload.get('logging_timestamp')

    new_input_payload = dict()
    new_input_payload['input'] = input_payload
    new_output_payload = dict()
    new_output_payload['output'] = output_payload
    payload_dict = {**new_input_payload, **new_output_payload}

    df = pd.DataFrame({
        'uid': [uid],
        'logging_timestamp': [logging_timestamp],
        'input_output_payloads': [payload_dict]
    })
    df.to_sql(name=table_name, schema=schema_name, con=make_mysql_connection(db_secret_name),
              dtype={'input_output_payloads': sqlalchemy.types.JSON}, if_exists='append', index=False)


def log_feature_importance_to_mysql(df, schema, db_conn):
    """"
    Logs feature importance scores to the feature_score table. This table is created by the stored procedure
    GenerateModelPerformanceTables.
    :param df: pandas dataframe containing all the columns expected by the feature_score table
    :param schema: name of the schema
    :param db_conn: database connection
    """
    db.write_dataframe_to_database(df, schema, 'feature_score', db_conn)


def log_model_scores_to_mysql(df, schema, db_conn):
    """"
    Logs model scores to the model_score table. This table is created by the stored procedure
    GenerateModelPerformanceTables.
    :param df: pandas dataframe containing all the columns expected by the model_score table
    :param schema: name of the schema
    :param db_conn: database connection
    """
    db.write_dataframe_to_database(df, schema, 'model_score', db_conn)
