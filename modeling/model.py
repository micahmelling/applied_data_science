import joblib
import os
import pandas as pd

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import cross_val_score

from modeling.config import MODELS_DIRECTORY, ENGINEERING_PARAM_GRID


def train_model(x_train, y_train, get_pipeline_function, model_name, model, param_space, n_trials, cv_times, scoring):
    """
    Trains a machine learning model, optimizes the hyperparameters, and saves the serialized model into the
    MODELS_DIRECTORY.
    :param x_train: x_train dataframe
    :param y_train: y_train series
    :param get_pipeline_function: callable that takes model to produce a scikit-learn pipeline
    :param model_name: name of the model
    :param model: instantiated model
    :param param_space: the distribution of hyperparameters to search over
    :param n_trials: number of trial to search for optimal hyperparameters
    :param cv_times: number of times to cross validation
    :param scoring: scoring method used for cross validation
    :returns: scikit-learn pipeline
    """
    print(f'training {model_name}...')
    pipeline = get_pipeline_function(model)
    param_space.update(ENGINEERING_PARAM_GRID)

    def _objective(config):
        pipeline.set_params(**config)
        cv_score = cross_val_score(pipeline, x_train, y_train, cv=cv_times, scoring=scoring).mean()
        tune.report(neg_log_loss=cv_score, done=True)

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    analysis = tune.run(
        _objective,
        mode='max',
        metric='neg_log_loss',
        config=param_space,
        num_samples=n_trials,
        scheduler=ASHAScheduler())

    pipeline.set_params(**analysis.best_config)
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, os.path.join(model_name, MODELS_DIRECTORY, f'{model_name}.pkl'), compress=3)
    return pipeline
