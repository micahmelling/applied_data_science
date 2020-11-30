import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from ds_helpers.db import connect_to_mysql
from ds_helpers.aws import get_secrets_manager_secret


IMAGES_PATH = 'output_images'


def generate_learning_curve(model, df, target, train_sizes_list, cv_times, scoring, file_path):
    y = df[target]
    x = df.drop(target, 1)
    train_sizes, train_scores, validation_scores = learning_curve(model, x, y, train_sizes=train_sizes_list,
                                                                  cv=cv_times, scoring=scoring, n_jobs=-1)
    train_scores = train_scores.mean(axis=1)
    validation_scores = validation_scores.mean(axis=1)
    plt.plot(train_sizes, train_scores, label='Training Error')
    plt.plot(train_sizes, validation_scores, label='Validation error')
    plt.ylabel('Score')
    plt.xlabel('Training Set Size')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(file_path)
    plt.clf()


if __name__ == "__main__":
    mysql_creds = get_secrets_manager_secret('churn-model-mysql')
    churn_df = pd.read_sql('''select * from churn_model.churn_data;''',
                           connect_to_mysql(mysql_creds, ssl_path='../data/rds-ca-2019-root.pem'))
    churn_df['churn'] = np.where(churn_df['churn'].str.startswith('y'), 1, 0)
    churn_df.drop(['id', 'meta__inserted_at', 'client_id', 'acquired_date'], 1, inplace=True)
    churn_df.fillna(value=0, inplace=True)
    churn_df = pd.get_dummies(churn_df, dummy_na=True)
    model = RandomForestClassifier(max_depth=10, min_samples_leaf=5)
    train_sizes_list = [0.2, 0.4, 0.6, 0.8, 1.0]
    generate_learning_curve(model, churn_df, 'churn', train_sizes_list, 3, 'roc_auc',
                            os.path.join(IMAGES_PATH, 'learning_curve.png'))
