import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, Trials, space_eval

from helpers.model_helpers import save_pipeline, save_cv_scores
from data.db import log_model_metadata


def train_model(x_train, y_train, get_pipeline_function, model_uid, model, param_space, iterations, cv_strategy,
                cv_scoring, static_param_space, db_schema_name=None, db_conn=None, log_to_db=False):
    """
    Trains a machine learning model, optimizes the hyperparameters, and saves the serialized model into the
    MODELS_DIRECTORY.
    :param x_train: x_train dataframe
    :param y_train: y_train series
    :param get_pipeline_function: callable that takes model to produce a scikit-learn pipeline
    :param model_uid: model uid
    :param model: instantiated model
    :param param_space: the distribution of hyperparameters to search over
    :param iterations: number of trial to search for optimal hyperparameters
    :param cv_strategy: cross validation strategy
    :param cv_scoring: scoring method used for cross validation
    :param static_param_space: parameter search space valid for all models (e.g. feature engineering)
    :param db_schema_name: database schema to log metrics to
    :param db_conn: database connection
    :param log_to_db: Boolean of whether or not to log model info to a database
    :returns: scikit-learn pipeline
    """
    print(f'training {model_uid}...')
    pipeline = get_pipeline_function(model)
    if static_param_space:
        param_space.update(static_param_space)

    cv_scores_df = pd.DataFrame()

    def _model_objective(params):
        pipeline.set_params(**params)
        score = cross_val_score(pipeline, x_train, y_train, cv=cv_strategy, scoring=cv_scoring, n_jobs=-1)

        temp_cv_scores_df = pd.DataFrame(score)
        temp_cv_scores_df = temp_cv_scores_df.reset_index()
        temp_cv_scores_df['index'] = 'fold_' + temp_cv_scores_df['index'].astype(str)
        temp_cv_scores_df = temp_cv_scores_df.T
        temp_cv_scores_df = temp_cv_scores_df.add_prefix('fold_')
        temp_cv_scores_df = temp_cv_scores_df.iloc[1:]
        temp_cv_scores_df['mean'] = temp_cv_scores_df.mean(axis=1)
        temp_cv_scores_df['std'] = temp_cv_scores_df.std(axis=1)
        temp_params_df = pd.DataFrame(params, index=list(range(0, len(params) + 1)))
        temp_cv_scores_df = pd.concat([temp_params_df, temp_cv_scores_df], axis=1)
        temp_cv_scores_df = temp_cv_scores_df.dropna()
        nonlocal cv_scores_df
        cv_scores_df = cv_scores_df.append(temp_cv_scores_df)

        return 1 - score.mean()

    trials = Trials()
    best = fmin(_model_objective, param_space, algo=tpe.suggest, max_evals=iterations, trials=trials)
    best_params = space_eval(param_space, best)

    cv_scores_df = cv_scores_df.sort_values(by=['mean'], ascending=False)
    cv_scores_df = cv_scores_df.reset_index(drop=True)
    cv_scores_df = cv_scores_df.reset_index()
    cv_scores_df = cv_scores_df.rename(columns={'index': 'ranking'})
    save_cv_scores(cv_scores_df, model_uid, 'cv_scores')

    pipeline.set_params(**best_params)
    pipeline.fit(x_train, y_train)
    save_pipeline(pipeline, model_uid, 'model')
    if log_to_db:
        log_model_metadata(model_uid, db_schema_name, db_conn)
    return pipeline


def calibrate_fitted_model(pipeline, model_uid, x_validation, y_validation, calibration_method='sigmoid'):
    """
    Trains a CalibratedClassiferCV using the fitted model from the pipeline as the base_estimator.

    :param pipeline: Fitted pipeline
    :param model_uid: model uid
    :param x_validation: x validation set
    :param y_validation: y validation set
    :param calibration_method: the calibration method to use; either sigmoid or isotonic; default is sigmoid
    :returns: sklearn pipeline with the model step as a fitted CalibratedClassifierCV
    """
    model = pipeline.named_steps['model']
    pipeline.steps.pop(len(pipeline) - 1)
    x_validation = pipeline.transform(x_validation)
    calibrated_model = CalibratedClassifierCV(model, cv='prefit', method=calibration_method)
    calibrated_model.fit(x_validation, y_validation)
    pipeline.steps.append(['model', calibrated_model])
    save_pipeline(pipeline, model_uid, 'model')
    return pipeline
