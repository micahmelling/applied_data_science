import time
import uuid
import os
import joblib
import numpy as np
import sentry_sdk
import flask_monitoringdashboard as dashboard
import pandas as pd

from rq import Queue
from random import random
from copy import deepcopy
from hashlib import sha256
from flask import Flask, session, request, jsonify, render_template, redirect, url_for, flash
from flask_talisman import Talisman
from sentry_sdk.integrations.flask import FlaskIntegration
from apscheduler.schedulers.background import BackgroundScheduler
from flask_swagger import swagger
from flasgger import Swagger

from data.redis_worker import conn
from app_settings import AWS_KEYS_SECRET, APP_SECRET, S3_BUCKET_NAME, SCHEMA_NAME, MODEL_1_PATH, MODEL_2_PATH, \
    MODEL_FEATURES, HEURISTIC_MODEL_PATH
from data.db import retrieve_app_config, get_client_ltv_table, get_training_data, get_password_for_username
from helpers.app_helpers import log_payload_to_s3, set_s3_keys, hit_config_refresh_endpoint, make_prediction, \
    convert_json_to_dataframe, ensure_json_matches_training_data, get_current_timestamp


def initialize_app():
    """
    Initializes our Flask application.
    - creates a Flask app object
    - sets AWS keys for uploading payloads to S3
    - retrieves and sets the application config
    - integrates with Sentry for error reporting
    - sets up a background scheduler to refresh teh config every 3,600 seconds
    - loads the trained model and sets it as a global object
    """
    app = Flask(__name__)

    set_s3_keys(AWS_KEYS_SECRET)

    config_dict = retrieve_app_config(SCHEMA_NAME)
    for key, value in config_dict.items():
        app.config[key] = value

    sentry_sdk.init(
        dsn=os.environ["SENTRY_DSN"],
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0
    )

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=hit_config_refresh_endpoint, trigger="interval", seconds=3_600)
    scheduler.start()

    global model_1
    model_1 = joblib.load(MODEL_1_PATH)
    global model_2
    if MODEL_2_PATH != "none":
        model_2 = joblib.load(MODEL_2_PATH)
    else:
        model_2 = None
    global heuristic_model
    heuristic_model = joblib.load(HEURISTIC_MODEL_PATH)

    return app


app = initialize_app()
# Talisman(app)
app.secret_key = APP_SECRET
dashboard.bind(app)
q = Queue(connection=conn)
swag = Swagger(app)


@app.route("/spec")
def spec():
    swag = swagger(app)
    swag['info']['version'] = "1.0"
    swag['info']['title'] = "Churn API"
    return jsonify(swag)


@app.before_request
def set_session_uid():
    """
    Sets a UID for each session
    """
    uid = str(uuid.uuid4())
    session["uid"] = uid


@app.route("/", methods=["POST", "GET"])
def home():
    """
    Home route that will confirm if the app is healthy
    """
    return "app is healthy"


@app.route("/health", methods=["POST", "GET"])
def health():
    """
    Health check endpoint that wil confirm if the app is healthy
    """
    return "app is healthy"


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == 'POST':
        form_submission = request.form
        username = str(form_submission['username'])
        password = str(form_submission['password'])
        hashed_password = sha256(password.encode('utf-8')).hexdigest()
        database_password = get_password_for_username(username)
        if hashed_password == database_password:
            session['logged_in'] = True
            return redirect(url_for('model_interface'))
        else:
            flash('Credentials are not valid. Please try again.')
    return render_template('login.html')


@app.route("/logout", methods=["POST", "GET"])
def logout():
    if request.method == 'POST':
        session['logged_in'] = False
        return redirect(url_for('login'))
    return render_template('logout.html')


@app.route('/model_interface', methods=['GET', 'POST'])
def model_interface():
    logged_in = session.get('logged_in', False)
    if logged_in:
        if request.method == 'POST':
            form_submission = request.form
            raw_clients = str(form_submission['clients'])
            client_list = raw_clients.split(',')
            client_list = [str(c) for c in client_list]
            model_df = get_training_data()
            model_df = model_df.loc[model_df['client_id'].isin(client_list)]
            if len(model_df) > 0:
                model_df.reset_index(inplace=True, drop=True)
                prediction_df = model_df[MODEL_FEATURES]
                predictions_df = pd.DataFrame(model_1.predict_proba(prediction_df)[:, 1], columns=['prediction'])
                predictions_df = pd.concat([model_df[['client_id']], predictions_df], axis=1)
                client_df = pd.DataFrame({'client_id': client_list,
                                          'prediction': ['client_id_not_found'] * len(client_list)})
                predictions_df = pd.concat([predictions_df, client_df], axis=0)
                predictions_df['client_id'] = predictions_df['client_id'].astype(str)
                predictions_df['client_id'] = predictions_df['client_id'].str.strip()
                predictions_df = predictions_df.drop_duplicates(subset=['client_id'], keep='first')
                return render_template('model_interface.html', predictions=predictions_df.to_html(header=True,
                                                                                                  index=False))
            else:
                return render_template('model_interface.html',
                                       predictions='None of the passed Client Ids could be found.')
        else:
            return render_template('model_interface.html', predictions='predictions will be rendered here')
    return redirect(url_for('login'))


@app.route("/config-refresh", methods=["POST", "GET"])
def config_refresh():
    """
    Endpoint to refresh the config.

    This invokes the retrieve_app_config function to query the relevant MySQL table with configuration values.
    """
    config_dict = retrieve_app_config(SCHEMA_NAME)
    for key, value in config_dict.items():
        app.config[key] = value
    return "config refresh hit"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to produce model predictions. Output is logged to S3.
    ---
    responses:
          Error:
            description: In the case of errors, a stock response will be returned
          Success:
            description: Otherwise, a prediction and associated diagnostics are returned
    """
    try:
        session["endpoint"] = "predict"
        response_start_time = time.time()
        input_data = request.json

        client_id = input_data.get("client_id", "000000")
        ltv_df = get_client_ltv_table()
        try:
            client_ltv = (ltv_df.loc[ltv_df["client_id"] == client_id])["ltv"].iloc[-1]
        except IndexError:
            client_ltv = 0

        input_data = ensure_json_matches_training_data(input_data, MODEL_FEATURES)
        input_df = convert_json_to_dataframe(input_data)
        prediction_1 = make_prediction(input_df, model_1)
        if model_2:
            prediction_2 = make_prediction(input_df, model_2)
            model_1_percentage = float(app.config.get("model_1_percentage", 0.50))
            if model_1_percentage < random():
                prediction = prediction_1
                model_used = "model_1"
            else:
                prediction = prediction_2
                model_used = "model_2"
        else:
            prediction = prediction_1
            prediction_2 = np.nan
            model_used = "model_1"
        heuristic_prediction = make_prediction(input_df, heuristic_model)
        processing_time = round(time.time() - response_start_time, 3)

        input_data["uid"] = session.get("uid")
        input_data["url"] = request.url
        input_data["endpoint"] = "predict"
        output = dict()
        output["prediction"] = prediction
        if prediction >= float(app.config.get("proba_cutoff", 0.75)):
            output["high_risk"] = "yes"
        else:
            output["high_risk"] = "no"

        output["response_time"] = processing_time
        output["ltv"] = client_ltv
        session["output"] = deepcopy(output)
        session["output"]["prediction_2"] = prediction_2
        session["output"]["model_1_path"] = MODEL_1_PATH
        session["output"]["model_2_path"] = MODEL_2_PATH
        session["output"]["model_used"] = model_used
        session["output"]["heuristic_prediction"] = heuristic_prediction
        session["input"] = input_data

        # job = q.enqueue_call(
        #     func=send_prediction_email, args=(output, ), result_ttl=5000
        # )

        print(output)
        return output
    except Exception as e:
        print(e)
        output = {
            "error": "app was not able to process request",
            "prediction": 0
        }
        return output
    finally:
        if session.get("endpoint") == "predict":
            uid = session.get("uid")
            input_payload = session.get("input")
            output_payload = session.get("output")
            output_payload["logging_timestamp"] = str(get_current_timestamp())
            output_payload["logging_epoch"] = time.time()
            # log_payload_to_s3(input_payload, output_payload, uid, S3_BUCKET_NAME)
            # record_client_id(client_id)


if __name__ == "__main__":
    app.run(debug=True)

    # do not log locally (later)
    # make sure sentry logs errors as is
    # ensure input data has correct dtypes
    # ensure local uses stage settings
    # need to add use_heuristic_percentage to config and use in app
    # need to fix paths so that everything is run from root
