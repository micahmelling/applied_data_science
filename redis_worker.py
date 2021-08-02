import os
import redis
import sentry_sdk

from ds_helpers.aws import get_secrets_manager_secret
from sentry_sdk.integrations.flask import FlaskIntegration
from rq import Worker, Queue, Connection


environment = os.environ['ENVIRONMENT']
if environment != 'local':
    sentry_sdk.init(
        dsn=get_secrets_manager_secret('churn-sentry-dsn').get('dsn'),
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0
    )


listen = ['default']
redis_env = os.getenv('REDIS_ENV')
if redis_env == 'use_local':
    redis_url = 'redis://localhost:6379'
else:
    redis_url = 'redis://redis:6379'
conn = redis.from_url(redis_url)


if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
