import pytest
import pandas as pd
import json

from app import app
from helpers.app_helpers import convert_json_to_dataframe, ensure_json_matches_training_data
from app_settings import MODEL_FEATURES


def load_json_file(path):
    with open(path, 'r') as file:
        return json.loads(file.read())


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


@pytest.fixture
def simple_json():
    return {'key': 'value'}


@pytest.fixture
def payload_1():
    return load_json_file('tests/payloads/test_payload_1.json')


@pytest.fixture
def ltv_df():
    return pd.DataFrame({
        'client_id': ['1010'],
        'ltv': [100]
    })


def test_valid_env():
    answer = 2 + 2
    assert answer == 4


def test_convert_dataframe_to_json(simple_json):
    df = convert_json_to_dataframe(simple_json)
    assert 'key' in df.columns


def test_ensure_json_matches_training_data(client, payload_1):
    json_object = ensure_json_matches_training_data(payload_1, MODEL_FEATURES)
    assert list(json_object.keys()) == MODEL_FEATURES


def test_app_responds(client):
    r = client.get('https://localhost/')
    assert b'app is healthy' in r.data


def test_predict_endpoint_responds(client, simple_json):
    r = client.post('https://localhost/predict', json=simple_json)
    assert b'prediction' in r.data


def test_get_ltv(client, payload_1):
    r = client.post('https://localhost/predict', json=payload_1)
    assert b'ltv' in r.data


def test_prediction(client, payload_1):
    r = client.post('https://localhost/predict', json=payload_1)
    prediction = r.json.get('prediction')
    assert 0 < prediction < 1
