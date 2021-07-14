import numpy as np
import pandas as pd
import os
import requests
import yagmail
import ast
import json
import datetime

from ds_helpers import aws
from app_settings import STAGE_URL, PROD_URL, EMAIL_SECRET


def log_payload_to_s3(input_payload, output_payload, uid, bucket_name):
    """
    Logs input and output payloads to S3 in a single object.

    :param input_payload: the input payload
    :param output_payload: the output payload
    :param uid: session UID
    :param bucket_name: the name of the S3 bucket to upload the payloads to
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


def ensure_json_matches_training_data(json_object, modeling_features_list):
    """
    Ensures the keys in the json_object include the features (i.e. modeling_features) used to train the model.

    :param json_object: json object:
    :param modeling_features_list: list of features used in modeling
    :returnsL json object
    """
    json_object = {k: json_object[k] for k in modeling_features_list if k in json_object}

    for feature in modeling_features_list:
        if feature not in json_object:
            json_object[feature] = np.nan

    index_map = {v: i for i, v in enumerate(modeling_features_list)}
    sorted(json_object.items(), key=lambda pair: index_map[pair[0]])
    return json_object


def convert_json_to_dataframe(json_object):
    """
    Converts a json object into a dataframe.

    :param json_object: json object
    :returns: pandas dataframe
    """
    df = pd.DataFrame.from_dict([json_object], orient='columns')
    return df


def set_s3_keys(secret_name):
    """
    Sets AWS keys that can interact with S3.

    :param secret_name: name of Secrets Manager secret
    """
    s3_keys_dict = aws.get_secrets_manager_secret(secret_name)
    access_key = s3_keys_dict.get('AWS_ACCESS_KEY_ID')
    secret_key = s3_keys_dict.get('AWS_SECRET_ACCESS_KEY')
    os.environ['AWS_ACCESS_KEY_ID'] = access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key


def hit_config_refresh_endpoint():
    """
    Hits the application's config-refresh endpoint to update the application's config.
    """
    if os.environ['ENVIRONMENT'] == 'local':
        url = 'http://127.0.0.1:5000/config-refresh'
    elif os.environ['ENVIRONMENT'] == 'stage':
        url = f'{STAGE_URL}/config-refresh'
    elif os.environ['ENVIRONMENT'] == 'prod':
        url = f'{PROD_URL}/config-refresh'
    else:
        raise ValueError('the ENVIRONMENT environment variable must be one of LOCAL, STAGE, or PROD')
    requests.get(url)


def make_prediction(df, model):
    """
    Makes a prediction on df using model.

    :param df: pandas dataframe
    :param model: fitted model with predict_proba method
    """
    return round(model.predict_proba(df)[0][1], 2)


def decode_list_str_elements(lst, decode_type='utf-8'):
    """
    Converts all elements to utf-8.

    :param lst: list
    :param decode_type: way to decode the string elements; default is utf-8
    :returns: list
    """
    lst = [i.decode(decode_type) for i in lst if type(i) == str]
    return lst


def count_list_item_occurrences(lst, lst_item):
    """
    Counts the occurrences of lst_item in lst.

    :param lst: list
    :param lst_item: valid object that can appear in a list
    :returns: int
    """
    return lst.count(lst_item)


def send_prediction_email(output_dict):
    """
    Sends email report with prediction information.

    :param output_dict: output dictionary produced by our app's predict function
    """
    email_dict = aws.get_secrets_manager_secret(EMAIL_SECRET)
    username = email_dict.get('username')
    password = email_dict.get('password')
    yag = yagmail.SMTP(username, password)
    recipients = email_dict.get('recipients')
    recipients = ast.literal_eval(recipients)
    client_id = output_dict.get('client_id', '000000')
    subject = 'Prediction Report for ' + str(client_id)
    contents = '<h3> Churn Prediction Report </h3>'
    contents = contents + json.dumps(output_dict)
    yag.send(to=recipients, subject=subject, contents=contents)


def get_current_timestamp():
    return datetime.datetime.now()
