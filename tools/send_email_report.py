import pandas as pd
import pytz
import datetime
import yagmail
import ast

from ds_helpers import aws, db


def get_start_timestamp(minute_lookback):
    now_cst = datetime.datetime.strptime(datetime.datetime.now(pytz.timezone('US/Central')).strftime(
        '%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    start_timestamp = now_cst - datetime.timedelta(minutes=minute_lookback)
    return start_timestamp


def query_logs_table(db_secret_name, start_timestamp):
    query = f'''
    select JSON_EXTRACT(input_output_payloads, "$.output.prediction") as prediction
    FROM churn_model.model_logs
    where logging_timestamp >= '{start_timestamp}';
    '''
    df = pd.read_sql(query, db.connect_to_mysql(aws.get_secrets_manager_secret(db_secret_name),
                                                ssl_path='data/rds-ca-2019-root.pem'))
    return df


def count_observation(df):
    return len(df)


def determine_should_email_be_sent(count, count_threshold):
    if count >= count_threshold:
        return True
    else:
        return False


def get_mean_prediction(df):
    return df['prediction'].astype(float).mean()


def send_email(email_secret, report_timestamp, mean_prediction, prediction_count):
    email_dict = aws.get_secrets_manager_secret(email_secret)
    username = email_dict.get('username')
    password = email_dict.get('password')
    yag = yagmail.SMTP(username, password)
    recipients = email_dict.get('recipients')
    recipients = ast.literal_eval(recipients)
    subject = f'Prediction Report since {report_timestamp}'
    contents = f'Number of predictions during window: {prediction_count}. \n Average prediction: {mean_prediction}.'
    yag.send(to=recipients, subject=subject, contents=contents)


def main(db_secret_name, minute_lookback, count_trigger_threshold, email_secret):
    start_timestamp = get_start_timestamp(minute_lookback)
    logs_df = query_logs_table(db_secret_name, start_timestamp)
    prediction_count = count_observation(logs_df)
    should_send_email = determine_should_email_be_sent(prediction_count, count_trigger_threshold)
    if should_send_email:
        mean_prediction = get_mean_prediction(logs_df)
        send_email(email_secret, start_timestamp, mean_prediction, prediction_count)


if __name__ == "__main__":
    main(db_secret_name='churn-model-mysql', minute_lookback=30, count_trigger_threshold=5, email_secret='churn-email')
