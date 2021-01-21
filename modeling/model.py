import joblib
import os
import pandas as pd

from tune_sklearn import TuneSearchCV
from sklearn.calibration import CalibratedClassifierCV

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
    search = TuneSearchCV(pipeline, param_distributions=param_space, n_trials=n_trials, scoring=scoring, cv=cv_times,
                          verbose=2, n_jobs=-1, search_optimization='hyperopt')
    search.fit(x_train, y_train)
    best_pipeline = search.best_estimator_
    cv_results = pd.DataFrame(search.cv_results_).sort_values(by=['rank_test_score'], ascending=True)
    joblib.dump(best_pipeline, os.path.join(model_name, MODELS_DIRECTORY, f'{model_name}.pkl'), compress=3)
    cv_results.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_cv_results.csv'), index=False)
    return best_pipeline


def calibrate_fitted_model(pipeline, x_validation, y_validation, calibration_method='sigmoid'):
    """
    Trains a CalibratedClassiferCV using the fitted model from the pipeline as the base_estimator.

    :param model: Fitted model
    :param x_validation: x validation set
    :param y_validation: y validation set
    :param calibration_method: the calibration method to use; either sigmoid or isotonic; default is sigmoid
    :returns: sklearn pipeline with the model step as a fitted CalibratedClassifierCV
    """
    model = pipeline.named_steps['model']
    pipeline.steps.pop(len(pipeline) - 1)
    x_validation = pipeline.transform(x_validation)
    calibrated_model = CalibratedClassifierCV(model, cv='prefit', method=calibration_method, n_jobs=-1)
    calibrated_model.fit(x_validation, y_validation)
    pipeline.steps.append(['model', calibrated_model])
    return pipeline
