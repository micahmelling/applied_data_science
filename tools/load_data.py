import pandas as pd

from time import sleep
from ds_helpers.db import connect_to_mysql, dynamically_create_ddl_and_execute, write_dataframe_to_database
from ds_helpers.aws import get_secrets_manager_secret


def main():
    mysql_creds = get_secrets_manager_secret('churn-model-mysql')
    db_conn = connect_to_mysql(mysql_creds, '../data/rds-ca-2019-root.pem')
    df = pd.read_csv('../data/site_churn_data.csv')
    dynamically_create_ddl_and_execute(df, 'churn_model', 'churn_data', db_conn)
    write_dataframe_to_database(df, 'churn_model', 'churn_data', db_conn)
    sleep(2)
    validation_df = pd.read_sql('''select * from churn_model.churn_data;''', db_conn)
    print(validation_df.head())


if __name__ == "__main__":
    main()
