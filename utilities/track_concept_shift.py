import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer

from data.db import make_mysql_connection
from helpers.model_helpers import create_train_test_split, create_x_y_split, FeaturesToDict, fill_missing_values
from helpers.utility_helpers import extract_model_uid_from_path, get_query_start_timestamp, extract_production_data, \
    recreate_data_used_for_training
from app_settings import MODEL_PATH
from modeling.config import FEATURE_DTYPE_MAPPING


def create_training_data(original_training_df, production_df):
    """
    Creates training data for building a model to determine if concept shift has occurred. Rows from production are
    the positive class.

    :param original_training_df: data used for training the model
    :param production_df: data seen in production
    :returns: x_train, x_test, y_train, y_test
    """
    original_training_df['target'] = 0
    production_df['target'] = 1
    training_df = pd.concat([original_training_df, production_df], axis=0)
    training_df.reset_index(inplace=True, drop=True)
    training_df.drop('acquired_date', 1, inplace=True)
    training_df = training_df.fillna(value=np.nan)
    x, y = create_x_y_split(training_df, 'target')
    x_train, x_test, y_train, y_test = create_train_test_split(x, y)
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """
    Trains a simple Random Forest model to determine if production data can be differentiated from training data.

    :param x_train: x_train
    :param y_train: y_train
    :returns: best trained pipeline
    """
    param_grid = {
        'model__max_depth': [5, 10, 15],
        'model__min_samples_leaf': [None, 3],
        'model__max_features': ['log2', 'sqrt']
    }

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', FunctionTransformer(fill_missing_values, validate=False,
                                        kw_args={'fill_value': 'unknown'})),
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer(sparse=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_transformer', numeric_transformer, selector(dtype_include='number')),
            ('categorical_transformer', categorical_transformer, selector(dtype_exclude='number'))
        ]
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier())
    ])

    search = GridSearchCV(pipeline, param_grid=param_grid, scoring='roc_auc', n_jobs=-1, cv=3)
    search.fit(x_train, y_train)
    best_pipeline = search.best_estimator_
    return best_pipeline


def evaluate_model(estimator, x_test, y_test):
    """
    Finds the ROC AUC of the model.

    :param estimator: fitted estimator
    :param x_test: x_test
    :param y_test: y_test
    """
    predictions = estimator.predict_proba(x_test)
    roc_auc = roc_auc_score(y_test, predictions[:, 1])
    return roc_auc


def determine_if_concept_shift_has_occurred(score, threshold):
    """
    If the model score is above a threshold, our model can differentiate between production and training data. Concept
    shift has occurred.

    :param score: model score on the test set
    :param threshold: threshold needed to pass to determine if concept shift has occurred
    :returns: Boolean
    """
    if score >= threshold:
        return True
    else:
        return False


def main(model_path, db_secret_name, scoring_threshold, model_features):
    """
    Determines if concept shift has occurred.

    :param model_path: path to the model
    :param db_secret_name: Secrets Manager secret with DB credentials
    :param scoring_threshold: threshold needed to pass to determine if concept shift has occurred
    :param model_features: features used for modeling
    """
    db_conn = make_mysql_connection(db_secret_name)
    model_uid = extract_model_uid_from_path(model_path)
    query_start_time = get_query_start_timestamp(model_uid, db_conn)
    production_df = extract_production_data(query_start_time, model_path, db_conn)
    original_training_df = recreate_data_used_for_training(model_uid, model_features)
    x_train, x_test, y_train, y_test = create_training_data(original_training_df, production_df)
    pipeline = train_model(x_train, y_train)
    model_score = evaluate_model(pipeline, x_test, y_test)
    shift_occurred = determine_if_concept_shift_has_occurred(model_score, scoring_threshold)
    insert_statement = f'''
        INSERT INTO churn_model.concept_shift (shift_occurred, metric_used, scoring_threshold, model_score, model_uid)
        VALUES ({shift_occurred}, 'roc_auc', {scoring_threshold}, {model_score}, '{model_uid}');
        '''
    db_conn.execute(insert_statement)
    with open(os.path.join('utilities', 'concept_shift_log.txt'), 'a') as file:
        file.write(f'For {model_uid}, has concept shift occurred: {shift_occurred}')


if __name__ == "__main__":
    main(
        model_path=MODEL_PATH,
        db_secret_name='churn-model-mysql',
        scoring_threshold=0.55,
        model_features=list(FEATURE_DTYPE_MAPPING.keys())
    )
