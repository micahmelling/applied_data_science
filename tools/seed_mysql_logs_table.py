import pandas as pd
import boto3
import os
import json
import glob
import sqlalchemy
import shutil

from ds_helpers import db, aws


def download_folder_from_s3(bucket_name, directory):
    """
    Downloads the contents of an entire folder from an S3 bucket into a local directory. If the remote directory is root,
    then the local directory name is constructed with the f-string f's3_{bucket_name}'. Otherwise, the name of the remote
    directory is also the name of the local directory.

    Parameters
    ----------
        bucket_name: name of the S3 bucket
        directory: name of the directory in the S3 bucket

    Returns
    -------
        this function does not return anything
    """
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)

    if directory == '/':
        directory_name = f's3_{bucket_name}'
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        for s3_object in bucket.objects.all():
            bucket.download_file(s3_object.key, os.path.join(directory_name, s3_object.key))
    else:
        for s3_object in bucket.objects.filter(Prefix=directory):
            if not os.path.exists(os.path.dirname(s3_object.key)):
                os.makedirs(os.path.dirname(s3_object.key))
            bucket.download_file(s3_object.key, s3_object.key)


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


def main(bucket_name, s3_directory, table_name, schema_name, db_secret_name):
    local_directory = f's3_{bucket_name}'
    try:
        download_folder_from_s3(bucket_name, s3_directory)
        insert_records(local_directory, table_name, schema_name, db_secret_name)
    finally:
        if os.path.exists(local_directory):
            shutil.rmtree(local_directory)


if __name__ == "__main__":
    main('churn-model-data-science-logs', '/', 'model_logs', 'churn_model', 'churn-model-mysql')
