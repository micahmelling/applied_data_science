import pandas as pd
import pathlib
import os
# import redis

from ds_helpers import db, aws
from cachetools import cached, TTLCache

from modeling.config import DB_SECRET_NAME
from app_settings import DATABASE_SECRET


path = pathlib.Path(__file__).parent.absolute()
# r = redis.StrictRedis()


@cached(cache=TTLCache(maxsize=1, ttl=86_400))
def get_training_data():
    """
    Retrieves the training data from MySQL.
    """
    return pd.read_sql('''select * from churn_model.churn_data;''',
                       db.connect_to_mysql(aws.get_secrets_manager_secret(DB_SECRET_NAME),
                                           ssl_path=os.path.join(path, 'rds-ca-2019-root.pem')))


def log_feature_importance_to_mysql(df, schema):
    """"
    Logs feature importance scores to the feature_score table. This table is created by the stored procedure
    GenerateModelPerformanceTables.

    :param df: pandas dataframe containing all the columns expected by the feature_score table
    :param schema: name of the schema
    """
    db.write_dataframe_to_database(df, schema, 'feature_score',
                                   db.connect_to_mysql(aws.get_secrets_manager_secret(DB_SECRET_NAME),
                                                       ssl_path=os.path.join(path, 'rds-ca-2019-root.pem')))


def log_model_scores_to_mysql(df, schema):
    """"
    Logs model scores to the model_score table. This table is created by the stored procedure
    GenerateModelPerformanceTables.

    :param df: pandas dataframe containing all the columns expected by the model_score table
    :param schema: name of the schema
    """
    db.write_dataframe_to_database(df, schema, 'model_score',
                                   db.connect_to_mysql(aws.get_secrets_manager_secret(DB_SECRET_NAME),
                                                       ssl_path=os.path.join(path, 'rds-ca-2019-root.pem')))


def retrieve_app_config(schema):
    """
    Retrieves the application configuration from a MySQL table.

    :return: dictionary with config keys and values
    """
    if os.environ['ENVIRONMENT'] == 'STAGE':
        table_name = 'stage_config'
    else:
        table_name = 'prod_config'
    query = f'''
    select config_key, config_value 
    from {schema}.{table_name}
    where meta__inserted_at = (select max(meta__inserted_at) from {schema}.{table_name})
    ;'''
    mysql_conn_dict = aws.get_secrets_manager_secret(DATABASE_SECRET)
    df = pd.read_sql(query, db.connect_to_mysql(mysql_conn_dict, ssl_path=os.path.join(path, 'rds-ca-2019-root.pem')))
    df = df.set_index('config_key')
    df_dict = df.to_dict().get('config_value')
    return df_dict


@cached(cache=TTLCache(maxsize=1, ttl=86_400))
def get_client_ltv_table():
    query = '''
    select client_id, ltv
    from churn_model.client_ltv;
    '''
    mysql_conn_dict = aws.get_secrets_manager_secret(DATABASE_SECRET)
    df = pd.read_sql(query, db.connect_to_mysql(mysql_conn_dict, ssl_path=os.path.join(path, 'rds-ca-2019-root.pem')))
    return df


def get_client_ids():
    query = '''
    select client_id
    from churn_model.churn_data;
    '''
    mysql_conn_dict = aws.get_secrets_manager_secret(DATABASE_SECRET)
    df = pd.read_sql(query, db.connect_to_mysql(mysql_conn_dict, ssl_path=os.path.join(path, 'rds-ca-2019-root.pem')))
    return df


def get_password_for_username(username):
    query = f'''
    select password
    from churn_model.app_credentials
    where username = '{username}';
    '''
    mysql_conn_dict = aws.get_secrets_manager_secret(DATABASE_SECRET)
    df = pd.read_sql(query, db.connect_to_mysql(mysql_conn_dict, ssl_path=os.path.join(path, 'rds-ca-2019-root.pem')))
    return df['password'][0]


# def record_client_id(client_id):
#     """
#     Pushes a new client_id to the client_ids Redis list.
#
#     :param client_id: client id
#     """
#     r.lpush('client_ids', client_id)
#
#
# def get_client_ids_already_seen():
#     """
#     Gets all of the values in the client_ids Redis list.
#
#     :returns: list
#     """
#     return r.lrange('client_ids', 0, -1)
