import pandas as pd
import os

from ds_helpers import aws


def log_payload_to_s3(input_payload, output_payload, uid, bucket_name):
    """

    """
    new_input_payload = dict()
    new_input_payload['input'] = input_payload
    new_output_payload = dict()
    new_output_payload['output'] = output_payload
    final_payload = str({**new_input_payload, **new_output_payload})
    with open(f'{uid}.json', 'w') as outfile:
        outfile.write(final_payload)
    aws.upload_file_to_s3(f'{uid}.json', bucket_name)
    os.remove(f'{uid}.json')


def convert_json_to_dataframe(json_object):
    """

    """
    df = pd.DataFrame.from_dict([json_object], orient='columns')
    return df


def set_s3_keys(secret_name):
    """

    """
    s3_keys_dict = aws.get_secrets_manager_secret(secret_name)
    access_key = s3_keys_dict.get('AWS_ACCESS_KEY_ID')
    secret_key = s3_keys_dict.get('AWS_SECRET_ACCESS_KEY')
    os.environ['AWS_ACCESS_KEY_ID'] = access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
