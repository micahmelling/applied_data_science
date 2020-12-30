import joblib
import os
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV

from modeling.config import MODELS_DIRECTORY, DIAGNOSTICS_DIRECTORY, ENGINEERING_PARAM_GRID


def train_model(x_train, y_train, get_pipeline_function, model_name, model, param_space, n_trials, cv_times, scoring):
    """
    Trains a machine learning model, optimizes the hyperparameters, saves the serialized model into the
    MODELS_DIRECTORY, and saves the cross validation results as a csv into the DIAGNOSTICS_DIRECTORY.

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
    search = RandomizedSearchCV(pipeline, param_distributions=param_space, n_iter=n_trials, scoring=scoring,
                                cv=cv_times, n_jobs=-1, verbose=10)
    search.fit(x_train, y_train)
    best_pipeline = search.best_estimator_
    cv_results = pd.DataFrame(search.cv_results_).sort_values(by=['rank_test_score'], ascending=False)
    joblib.dump(best_pipeline, os.path.join(model_name, MODELS_DIRECTORY, f'{model_name}.pkl'), compress=3)
    cv_results.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_cv_results.csv'), index=False)
    return best_pipeline
