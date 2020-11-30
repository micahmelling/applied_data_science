import uuid
import os
import sentry_sdk

from flask import Flask, session, request
from sentry_sdk.integrations.flask import FlaskIntegration

from helpers.app_helpers import log_payload_to_s3, set_s3_keys


app = Flask(__name__)
app.secret_key = os.environ['CHURN_APP_SECRET']
app.config.from_pyfile('app_config.py')


sentry_sdk.init(
    dsn=os.environ['SENTRY_DSN'],
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)


@app.before_first_request
def before_first_request_func():
    set_s3_keys('churn-api-s3-keys')


@app.before_request
def before_request_setup():
    uid = str(uuid.uuid4())
    session['uid'] = uid


@app.after_request
def after_request_func(response):
    if session.get('endpoint') == 'predict':
        uid = session.get('uid')
        input_payload = session.get('input')
        output_payload = session.get('output')
        log_payload_to_s3(input_payload, output_payload, uid, 'churn-model-data-science-logs')
    return response


@app.route('/', methods=['POST', 'GET'])
def home():
    """ home route that wil confirm if the app is healthy """
    return 'app is healthy'


@app.route('/health', methods=['POST', 'GET'])
def health():
    """ health endpoint that wil confirm if the app is healthy """
    return 'app is healthy'


@app.route('/predict', methods=['POST'])
def predict():
    """ predict endpoint to produce model predictions """
    try:
        input_data = request.json
        input_data['uid'] = session.get('uid')
        input_data['url'] = request.url
        input_data['endpoint'] = 'predict'
        input_data['mapping'] = app.config.get('MAPPING')
        output = input_data
        session['endpoint'] = 'predict'
        session['input'] = input_data
        session['output'] = output
        print(output)
        return output
    except Exception as e:
        print(e)
        return {'app unable to respond to input; please check the payload'}


if __name__ == "__main__":
    app.run(debug=True)
