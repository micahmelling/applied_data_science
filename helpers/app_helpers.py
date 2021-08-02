import pandas as pd
import requests
import datetime

from app_settings import URL


def convert_json_to_dataframe(json_object):
    """
    Converts a json object into a dataframe.

    :param json_object: json object
    :returns: pandas dataframe
    """
    df = pd.DataFrame.from_dict([json_object], orient='columns')
    return df


def hit_config_refresh_endpoint():
    """
    Hits the application's config-refresh endpoint to update the application's config.
    """
    requests.get(f'https://{URL}/config-refresh')


def make_prediction(df, model):
    """
    Makes a prediction on df using model.

    :param df: pandas dataframe
    :param model: fitted model with predict_proba method
    """
    return round(model.predict_proba(df)[0][1], 2)


def get_current_timestamp():
    """
    Gets and returns the current UTC timestamp.
    """
    return datetime.datetime.now()
