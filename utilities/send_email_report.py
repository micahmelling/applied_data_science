import pandas as pd
import pytz
import datetime
import yagmail
import ast

from data.db import make_mysql_connection


def get_start_timestamp(minute_lookback):
    """
    Gets a timestamp minute_lookback minutes ago.

    :param minute_lookback: number of minutes ago
    :returns: timestamp
    """
    now_cst = datetime.datetime.strptime(datetime.datetime.now(pytz.timezone('US/Central')).strftime(
        '%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    start_timestamp = now_cst - datetime.timedelta(minutes=minute_lookback)
    return start_timestamp


def query_logs_table(db_secret_name, start_timestamp):
    """
    Queries table of API logs.

    :param db_secret_name: name of the Secrets Manager secret with the database credentials
    :param start_timestamp: timestamp for which to pull logs starting at
    :returns: pandas dataframe
    """
    query = f'''
    select JSON_EXTRACT(input_output_payloads, "$.output.prediction") as prediction
    FROM churn_model.model_logs
    where logging_timestamp >= '{start_timestamp}';
    '''
    df = pd.read_sql(query, make_mysql_connection(db_secret_name))
    return df


def count_observation(df):
    """
    Counts the number of rows in a dataframe.

    :param df: pandas dataframe
    :returns: int
    """
    return len(df)


def determine_should_email_be_sent(count, count_threshold):
    """
    Produces relevant Boolean if count is above count_threshold.

    :param count: int
    :param count_threshold: int
    :returns: Boolean
    """
    if count >= count_threshold:
        return True
    else:
        return False


def get_mean_prediction(df):
    """
    Finds the mean of the prediction column.

    :param df: pandas dataframe
    :returns: float
    """
    return df['prediction'].astype(float).mean()


def send_email(email_secret, report_timestamp, mean_prediction, prediction_count):
    """
    Sends an email reporting the mean prediction value and the number of predictions.

    :param email_secret: Secrets Manager secret with yagmail credentials
    :param report_timestamp: timestamp the report was produced
    :param mean_prediction: mean prediction value
    :param prediction_count: number of predictions made
    """
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
    """
    Sends an email report if the number of predictions in a certain window exceeds a set threshold.

    :param db_secret_name: name of the Secrets Manager secret with the database credentials
    :param minute_lookback: number of minutes ago to start looking at data
    :param count_trigger_threshold: threshold for predictions to trigger email
    :param email_secret: Secrets Manager secret with yagmail credentials
    """
    start_timestamp = get_start_timestamp(minute_lookback)
    logs_df = query_logs_table(db_secret_name, start_timestamp)
    prediction_count = count_observation(logs_df)
    should_send_email = determine_should_email_be_sent(prediction_count, count_trigger_threshold)
    if should_send_email:
        mean_prediction = get_mean_prediction(logs_df)
        send_email(email_secret, start_timestamp, mean_prediction, prediction_count)


if __name__ == "__main__":
    main(
        db_secret_name='churn-model-mysql',
        minute_lookback=30,
        count_trigger_threshold=50,
        email_secret='churn-email'
    )
