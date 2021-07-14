import boto3
import os
import glob
import json

from time import sleep
from decimal import Decimal


def download_folder_from_s3(bucket_name, directory):
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


def create_dynamo_table(table_name, key_field):
    dynamodb = boto3.resource('dynamodb')
    dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': key_field,
                'KeyType': 'HASH'
            },
        ],
        AttributeDefinitions=[
            {
                'AttributeName': key_field,
                'AttributeType': 'S'
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10
        }
    )
    sleep(5)


def insert_records(directory, table_name):
    dynamodb = boto3.resource('dynamodb')
    dynamo_table = dynamodb.Table(table_name)
    for path in glob.glob(f'{directory}/*'):
        with open(path, 'r') as file:
            payload_dict = json.loads(json.dumps(json.loads(file.read().replace("\'", "\""))), parse_float=Decimal)
            uid = payload_dict.get('input').get('uid')
            uid_dict = dict()
            uid_dict['uid'] = uid
            uid_dict.update(payload_dict)
            dynamo_table.put_item(Item=uid_dict)


def main(s3_bucket, directory, dynamo_table, key_field, create_table=True):
    if create_table:
        download_folder_from_s3(s3_bucket, directory)
    create_dynamo_table(dynamo_table, key_field)
    insert_records(f's3_{s3_bucket}', dynamo_table)


if __name__ == "__main__":
    main('churn-model-data-science-logs', '/', 'churn_logs', 'uid')
