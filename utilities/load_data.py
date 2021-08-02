import pandas as pd

from time import sleep
from ds_helpers.db import dynamically_create_ddl_and_execute, write_dataframe_to_database

from data.db import make_mysql_connection


def main():
    """
    Loads the csv churn data into a MySQL table.
    """
    db_conn = make_mysql_connection('churn-model-mysql')
    df = pd.read_csv('data/site_churn_data.csv')
    dynamically_create_ddl_and_execute(df, 'churn_model', 'churn_data', db_conn)
    write_dataframe_to_database(df, 'churn_model', 'churn_data', db_conn)
    sleep(2)
    validation_df = pd.read_sql('''select * from churn_model.churn_data;''', db_conn)
    print(validation_df.head())


if __name__ == "__main__":
    main()
