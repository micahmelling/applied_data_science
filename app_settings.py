import os

from ds_helpers.aws import get_secrets_manager_secret


ENVIRONMENT = os.environ['ENVIRONMENT']
MODEL_PATH = os.path.join('modeling', 'random_forest_202107252202562870520500', 'model', 'model.pkl')
OUTPUT_LOGS_TABLE_NAME = 'model_logs'
DB_SCHEMA = 'churn_model'
SENTRY_DSN = get_secrets_manager_secret('churn-sentry-dsn').get('dsn')
FLASK_SECRET = get_secrets_manager_secret('churn-app-flask-secret').get('secret')
OUTPUT_LOGS_S3_BUCKET_NAME = 'churn-model-data-science-logs'

if ENVIRONMENT == 'local':
    URL = '127.0.0.1:5000'
    DATABASE_SECRET = 'stage-churn-model-svc-mysql'
elif ENVIRONMENT == 'stage':
    URL = 'stage-url'
    DATABASE_SECRET = 'stage-churn-model-svc-mysql'
elif ENVIRONMENT == 'prod':
    URL = 'prod-url'
    DATABASE_SECRET = 'prod-churn-model-svc-mysql'
else:
    raise Exception(f'ENVIRONMENT must be one of local, stage, or prod. {ENVIRONMENT} was passed.')
