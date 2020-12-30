import pandas as pd

from ds_helpers import db, aws

from modeling.config import DB_SECRET_NAME


def get_training_data():
    """
    Retrieves the training data from MySQL.
    """
    return pd.read_sql('''select * from churn_model.churn_data;''',
                       db.connect_to_mysql(aws.get_secrets_manager_secret(DB_SECRET_NAME),
                                           ssl_path='../data/rds-ca-2019-root.pem'))


def log_feature_importance_to_mysql(df, schema):
    """"
    Logs feature importance scores to the feature_score table. This table is created by the stored procedure
    GenerateModelPerformanceTables.

    :param df: pandas dataframe containing all the columns expected by the feature_score table
    :param schema: name of the schema
    """
    db.write_dataframe_to_database(df, schema, 'feature_score',
                                   db.connect_to_mysql(aws.get_secrets_manager_secret(DB_SECRET_NAME),
                                                       ssl_path='../data/rds-ca-2019-root.pem'))


def log_model_scores_to_mysql(df, schema):
    """"
    Logs model scores to the model_score table. This table is created by the stored procedure
    GenerateModelPerformanceTables.

    :param df: pandas dataframe containing all the columns expected by the model_score table
    :param schema: name of the schema
    """
    db.write_dataframe_to_database(df, schema, 'model_score',
                                   db.connect_to_mysql(aws.get_secrets_manager_secret(DB_SECRET_NAME),
                                                       ssl_path='../data/rds-ca-2019-root.pem'))
