import joblib
import os
import pandas as pd
import numpy as np
import json

from tune_sklearn import TuneSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from ultraopt.hdl import layering_config
from ultraopt import fmin
from ultraopt.multi_fidelity import HyperBandIterGenerator
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from modeling.config import MODELS_DIRECTORY, DIAGNOSTICS_DIRECTORY, ENGINEERING_PARAM_GRID


# def train_model(x_train, y_train, get_pipeline_function, model_name, model, param_space, n_trials, cv_times, scoring):
#     """
#     Trains a machine learning model, optimizes the hyperparameters, saves the serialized model into the
#     MODELS_DIRECTORY, and saves the cross validation results as a csv into the DIAGNOSTICS_DIRECTORY.
#     :param x_train: x_train dataframe
#     :param y_train: y_train series
#     :param get_pipeline_function: callable that takes model to produce a scikit-learn pipeline
#     :param model_name: name of the model
#     :param model: instantiated model
#     :param param_space: the distribution of hyperparameters to search over
#     :param n_trials: number of trial to search for optimal hyperparameters
#     :param cv_times: number of times to cross validation
#     :param scoring: scoring method used for cross validation
#     :returns: scikit-learn pipeline
#     """
#     print(f'training {model_name}...')
#     pipeline = get_pipeline_function(model)
#     param_space.update(ENGINEERING_PARAM_GRID)
#     search = RandomizedSearchCV(pipeline, param_distributions=param_space, n_iter=n_trials, scoring=scoring,
#                                 cv=cv_times, verbose=2, n_jobs=-1)
#     search.fit(x_train, y_train)
#     best_pipeline = search.best_estimator_
#     cv_results = pd.DataFrame(search.cv_results_).sort_values(by=['rank_test_score'], ascending=True)
#     joblib.dump(best_pipeline, os.path.join(model_name, MODELS_DIRECTORY, 'model.pkl'), compress=3)
#     cv_results.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'cv_results.csv'), index=False)
#     return best_pipeline


# def train_model(x_train, y_train, get_pipeline_function, model_name, model, param_space, n_trials, cv_times, scoring):
#     """
#     Trains a machine learning model, optimizes the hyperparameters, saves the serialized model into the
#     MODELS_DIRECTORY, and saves the cross validation results as a csv into the DIAGNOSTICS_DIRECTORY.
#     :param x_train: x_train dataframe
#     :param y_train: y_train series
#     :param get_pipeline_function: callable that takes model to produce a scikit-learn pipeline
#     :param model_name: name of the model
#     :param model: instantiated model
#     :param param_space: the distribution of hyperparameters to search over
#     :param n_trials: number of trial to search for optimal hyperparameters
#     :param cv_times: number of times to cross validation
#     :param scoring: scoring method used for cross validation
#     :returns: scikit-learn pipeline
#     """
#     print(f'training {model_name}...')
#     pipeline = get_pipeline_function(model)
#
#     def _evaluate(config):
#         local_pipe = pipeline.set_params(**layering_config(config))
#         return 1 - float(cross_val_score(local_pipe, x_train, y_train, scoring=scoring, cv=cv_times, n_jobs=-1).mean())
#
#     hb = HyperBandIterGenerator(min_budget=1/4, max_budget=1, eta=2)
#     result = fmin(eval_func=_evaluate, config_space=param_space, optimizer="ETPE", n_iterations=n_trials,
#                   multi_fidelity_iter_generator=hb)
#     best_config = result.best_config
#     pipeline.set_params(**best_config)
#     pipeline.fit(x_train, y_train)
#     joblib.dump(pipeline, os.path.join(model_name, MODELS_DIRECTORY, f'{model_name}.pkl'), compress=3)
#     return pipeline


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

    def _evaluate(cfg):
        local_pipe = pipeline.set_params(**cfg)
        return 1 - float(cross_val_score(local_pipe, x_train, y_train, scoring=scoring, cv=cv_times, n_jobs=1).mean())

    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": n_trials,
        "cs": param_space,
        "deterministic": "true",
    })

    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42), tae_runner=_evaluate)
    value = smac.get_tae_runner().run(param_space.get_default_configuration(), 1)[1]
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    output_dir = smac.output_dir
    with open(os.path.join(output_dir, 'traj_aclib2.json'), 'r') as file:
        json_strs = file.readlines()

    lowest_cost = 100_000_000
    best_parameters = []
    for json_str in json_strs:
        json_str = json_str.replace("'", '')
        local_json = json.loads(json_str)
        cost = local_json.get('cost')
        if cost < lowest_cost:
            lowest_cost = cost
            best_parameters = local_json.get('incumbent')

    parameter_dict = dict()
    for parameter in best_parameters:
        param_split = parameter.split('=')
        parameter_dict[param_split[0]] = param_split[1]

    for k, v in parameter_dict.items():
        try:
            parameter_dict[k] = float(v)
        except:
            pass

    pipeline.set_params(**parameter_dict)
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, os.path.join(model_name, MODELS_DIRECTORY, f'{model_name}.pkl'), compress=3)
    return pipeline


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
