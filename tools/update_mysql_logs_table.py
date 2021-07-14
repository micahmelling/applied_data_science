import pandas as pd
import boto3
import os
import json
import glob
import sqlalchemy
import shutil

from time import sleep
from ds_helpers import db, aws


def get_most_recent_db_insert(db_secret_name: str) -> str:
    """
    Gets the most recent logging_timestamp inserted into the churn_model.model_logs table

    :param db_secret_name: name of the Secrets Manager secret for DB credentials
    :returns: most recent logging_timestamp
    """
    query = '''
    select max(logging_timestamp) as max_insert
    from churn_model.model_logs;
    '''
    df = pd.read_sql(query, db.connect_to_mysql(aws.get_secrets_manager_secret(db_secret_name),
                                                ssl_path='data/rds-ca-2019-root.pem'))
    max_insert = df['max_insert'][0]
    return max_insert


def run_athena_query(client, start_timestamp):
    response = client.start_query_execution(
        QueryString=f'''select input.uid from ds_churn_logs where output.logging_timestamp >= TIMESTAMP '{start_timestamp}';''',
        QueryExecutionContext={
            'Database': 'churn_model'
        },
        ResultConfiguration={
            'OutputLocation': 's3://athena-churn-results/'
        }
    )
    return response


def get_athena_file_name(client, execution_response):
    execution_id = execution_response['QueryExecutionId']
    state = 'RUNNING'
    while state != 'SUCCEEDED':
        response = client.get_query_execution(QueryExecutionId=execution_id)
        state = response['QueryExecution']['Status']['State']
        if state == 'SUCCEEDED':
            s3_file_name = response['QueryExecution']['ResultConfiguration']['OutputLocation']
            s3_file_name = s3_file_name.split("/", 3)[3]
            return s3_file_name
        sleep(1)


def get_uids_to_download(file_name):
    aws.download_file_from_s3(file_name, 'athena-churn-results')
    df = pd.read_csv(file_name)
    uids = df['uid'].tolist()
    os.remove(file_name)
    return uids


def download_new_payloads(uids, local_directory):
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
    for uid in uids:
        aws.download_file_from_s3(f'{uid}.json', 'churn-model-data-science-logs')
        os.replace(f'{uid}.json', os.path.join(local_directory, f'{uid}.json'))


def insert_records(directory, table_name, schema_name, db_secret_name):
    main_df = pd.DataFrame()
    for path in glob.glob(f'{directory}/*'):
        with open(path, 'r') as file:
            payload_dict = json.loads(file.read().replace("\'", "\""))
            uid = payload_dict.get('input').get('uid')
            logging_timestamp = payload_dict.get('output').get('logging_timestamp')
            temp_df = pd.DataFrame({'uid': [uid],
                                    'logging_timestamp': [logging_timestamp],
                                    'input_output_payloads': [payload_dict]})
            main_df = main_df.append(temp_df)
    main_df.to_sql(name=table_name, schema=schema_name, con=db.connect_to_mysql(
        aws.get_secrets_manager_secret(db_secret_name),
        ssl_path='data/rds-ca-2019-root.pem'), dtype={'input_output_payloads': sqlalchemy.types.JSON},
                   if_exists='append', index=False)


def main(local_payloads_directory, db_secret_name, table_name, schema_name):
    try:
        client = boto3.client('athena')
        max_logging_timestamp = get_most_recent_db_insert(db_secret_name)
        athena_response = run_athena_query(client, max_logging_timestamp)
        athena_results_file_name = get_athena_file_name(client, athena_response)
        uids_to_download = get_uids_to_download(athena_results_file_name)
        download_new_payloads(uids_to_download, local_payloads_directory)
        insert_records(local_payloads_directory, table_name, schema_name, db_secret_name)
    finally:
        pass
        if os.path.exists(local_payloads_directory):
            shutil.rmtree(local_payloads_directory)


if __name__ == "__main__":
    main('temp_payloads', 'churn-model-mysql', 'model_logs', 'churn_model')
