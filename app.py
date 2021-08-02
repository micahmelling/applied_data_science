import time
import uuid
import joblib
import sentry_sdk
import flask_monitoringdashboard as dashboard
import pandas as pd

from rq import Queue
from copy import deepcopy
from hashlib import sha256
from flask import Flask, session, request, render_template, redirect, url_for, flash
from flask_talisman import Talisman
from sentry_sdk.integrations.flask import FlaskIntegration
from apscheduler.schedulers.background import BackgroundScheduler
from flasgger import Swagger
from ds_helpers.aws import log_payload_to_s3

from redis_worker import conn
from app_settings import ENVIRONMENT, FLASK_SECRET, OUTPUT_LOGS_S3_BUCKET_NAME, MODEL_PATH, \
    DATABASE_SECRET, OUTPUT_LOGS_TABLE_NAME, SENTRY_DSN, DB_SCHEMA, URL
from data.db import retrieve_app_config, get_client_ltv_table, get_training_data, get_hashed_password_for_username, \
    log_payloads_to_mysql, make_mysql_connection
from helpers.app_helpers import hit_config_refresh_endpoint, make_prediction, convert_json_to_dataframe, \
    get_current_timestamp


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

    if ENVIRONMENT != 'local':
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[FlaskIntegration()],
            traces_sample_rate=1.0
        )

    config_dict = retrieve_app_config(DB_SCHEMA, make_mysql_connection(DATABASE_SECRET), ENVIRONMENT)
    for key, value in config_dict.items():
        app.config[key] = value

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=hit_config_refresh_endpoint, trigger="interval", seconds=3_600)
    scheduler.start()

    global model
    model = joblib.load(MODEL_PATH)

    return app


app = initialize_app()
Talisman(app)
app.secret_key = FLASK_SECRET
dashboard.bind(app)
swag = Swagger(app)
q = Queue(connection=conn)


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
    """
    Login endpoint for the model user interface.
    """
    if request.method == 'POST':
        form_submission = request.form
        username = str(form_submission['username'])
        password = str(form_submission['password'])
        hashed_password = sha256(password.encode('utf-8')).hexdigest()
        database_password = get_hashed_password_for_username(username, make_mysql_connection(DATABASE_SECRET))
        if hashed_password == database_password:
            session['logged_in'] = True
            return redirect(url_for('model_interface'))
        else:
            flash('Credentials are not valid. Please try again.')
    return render_template('login.html')


@app.route("/logout", methods=["POST", "GET"])
def logout():
    """
    Logout endpoint for the model user interface.
    """
    if request.method == 'POST':
        session['logged_in'] = False
        return redirect(url_for('login'))
    return render_template('logout.html')


@app.route('/model_interface', methods=['GET', 'POST'])
def model_interface():
    """
    Model user interface to render predictions in HTML.
    """
    logged_in = session.get('logged_in', False)
    if logged_in:
        if request.method == 'POST':
            form_submission = request.form
            raw_clients = str(form_submission['clients'])
            client_list = raw_clients.split(',')
            client_list = [str(c) for c in client_list]
            model_df = get_training_data(make_mysql_connection(DATABASE_SECRET))
            model_df = model_df.loc[model_df['client_id'].isin(client_list)]
            if len(model_df) > 0:
                model_df.reset_index(inplace=True, drop=True)
                predictions_df = pd.DataFrame(model.predict_proba(model_df)[:, 1], columns=['prediction'])
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
    Endpoint to refresh the config. This invokes the retrieve_app_config function to query the relevant MySQL table with
    configuration values.
    """
    config_dict = retrieve_app_config(DB_SCHEMA, make_mysql_connection(DATABASE_SECRET), ENVIRONMENT)
    for key, value in config_dict.items():
        app.config[key] = value
    return "config refresh hit"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to produce model predictions. Output is logged to S3.
    """
    try:
        session["endpoint"] = "predict"
        response_start_time = time.time()
        input_data = request.json

        ltv_df = get_client_ltv_table(make_mysql_connection(DATABASE_SECRET))
        client_id = input_data.get("client_id", "000000")
        try:
            client_ltv = (ltv_df.loc[ltv_df["client_id"] == client_id])["ltv"].iloc[-1]
        except IndexError:
            client_ltv = 0

        input_df = convert_json_to_dataframe(input_data)
        prediction = make_prediction(input_df, model)
        if prediction >= float(app.config.get("proba_cutoff", 0.75)):
            high_risk = "yes"
        else:
            high_risk = "no"

        processing_time = round(time.time() - response_start_time, 3)

        input_data["uid"] = session.get("uid")
        input_data["url"] = request.url
        input_data["endpoint"] = "predict"
        output = dict()
        output["prediction"] = prediction
        output["high_risk"] = high_risk
        output["response_time"] = processing_time
        output["ltv"] = client_ltv
        session["output"] = deepcopy(output)
        session["input"] = input_data

        print(output)
        return output
    except Exception as exception:
        print(exception)
        sentry_sdk.capture_exception(exception)
        output = {
            "error": "app was not able to process request",
            "prediction": 0
        }
        return output
    finally:
        if ENVIRONMENT != "local":
            uid = session.get("uid")
            input_payload = session.get("input")
            output_payload = session.get("output", {})
            output_payload["logging_timestamp"] = str(get_current_timestamp())
            output_payload["logging_epoch"] = time.time()
            job = q.enqueue_call(
                func=log_payload_to_s3, args=(input_payload, output_payload, uid, OUTPUT_LOGS_S3_BUCKET_NAME, ),
                result_ttl=20
            )
            job = q.enqueue_call(
                func=log_payloads_to_mysql, args=(input_payload, output_payload, OUTPUT_LOGS_TABLE_NAME, DB_SCHEMA,
                                                  DATABASE_SECRET, ), result_ttl=20
            )
