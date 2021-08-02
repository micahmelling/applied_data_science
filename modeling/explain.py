import shap
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp

from statistics import mean
from sklearn.inspection import permutation_importance
from copy import deepcopy
from functools import partial
from dtreeviz.trees import dtreeviz
from sklearn.inspection import plot_partial_dependence
from PyALE import ale

from helpers.model_helpers import make_directories_if_not_exists, find_non_dummied_columns, \
    transform_data_with_pipeline, determine_if_name_in_object
from data.db import log_feature_importance_to_mysql

plt.switch_backend('Agg')


def _run_shap_explainer(x_df, explainer, boosting_model, use_kernel, nsamples_kernel=500):
    """
    Runs the SHAP explainer on a dataframe.

    :param x_df: x dataframe
    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param use_kernel: Boolean of whether or not to use Kernel SHAP, which mostly makes sense when we are using a
    CalibratedClassifierCV
    :param nsamples_kernel: number of samples to use when employing the kernel explainer
    """
    if boosting_model:
        if use_kernel:
            return explainer.shap_values(x_df, nsamples=nsamples_kernel, check_additivity=False)
        else:
            return explainer.shap_values(x_df, check_additivity=False)
    else:
        if use_kernel:
            return explainer.shap_values(x_df, nsamples=nsamples_kernel, check_additivity=False)[1]
        else:
            return explainer.shap_values(x_df, check_additivity=False)[1]


def _run_parallel_shap_explainer(x_df, explainer, boosting_model, use_kernel):
    """
    Splits x_df into evenly-split partitions based on the number of CPU available on the machine. Then, the SHAP
    explainer object is run in parallel on each subset of x_df. The results are then combined into a single object.

    :param x_df: x dataframe
    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param use_kernel: Boolean of whether or not to use Kernel SHAP, which mostly makes sense when we are using a
    CalibratedClassifierCV
    """
    array_split = np.array_split(x_df, mp.cpu_count())
    shap_fn = partial(_run_shap_explainer, explainer=explainer, boosting_model=boosting_model, use_kernel=use_kernel)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.map(shap_fn, array_split)
    result = np.concatenate(result)
    return result


def _get_shap_expected_value(explainer, boosting_model):
    """
    Extracts a SHAP Explainer's expected value.

    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :returns: int
    """
    if boosting_model:
        expected_value = explainer.expected_value[0]
    else:
        try:
            expected_value = explainer.expected_value[1]
        except IndexError:
            expected_value = explainer.expected_value[0]
    return expected_value


def _produce_raw_shap_values(model, model_uid, x_df, calibrated, boosting_model, use_kernel):
    """
    Produces the raw shap values for every observation in the test set. A dataframe of the shap values is saved locally
    as a csv. The shap expected value is extracted and save locally in a csv.

    :param model: fitted model
    :param model_uid: model uid
    :param x_df: x dataframe
    :param calibrated: boolean of whether or not the model is a CalibratedClassifierCV; the default is False
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param use_kernel: Boolean of whether or not to use Kernel SHAP, which mostly makes sense when we are using a
    CalibratedClassifierCV
    :returns: numpy array
    """
    if calibrated:
        if use_kernel:
            explainer = shap.KernelExplainer(model.predict_proba, x_df.iloc[:50, :])
            shap_values = _run_parallel_shap_explainer(x_df, explainer, boosting_model, True)
            shap_expected_value = _get_shap_expected_value(explainer, boosting_model)
        else:
            shap_values_list = []
            shap_expected_list = []
            for calibrated_classifier in model.calibrated_classifiers_:
                explainer = shap.TreeExplainer(calibrated_classifier.base_estimator)
                shap_values = _run_parallel_shap_explainer(x_df, explainer, boosting_model, False)
                shap_expected_value = _get_shap_expected_value(explainer, boosting_model)
                shap_values_list.append(shap_values)
                shap_expected_list.append(shap_expected_value)
            shap_values = np.array(shap_values_list).sum(axis=0) / len(shap_values_list)
            shap_expected_value = mean(shap_expected_list)
        shap_df = pd.DataFrame(shap_values, columns=list(x_df))
        shap_df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'shap', 'shap_values.csv'), index=False)
        shap_expected_value = pd.DataFrame({'expected_value': [shap_expected_value]})
        shap_expected_value.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'shap', 'shap_expected.csv'),
                                   index=False)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = _run_parallel_shap_explainer(x_df, explainer, boosting_model, False)
        shap_df = pd.DataFrame(shap_values, columns=list(x_df))
        shap_df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'shap', 'shap_values.csv'), index=False)
        shap_expected_value = _get_shap_expected_value(explainer, boosting_model)
        shap_expected_value = pd.DataFrame({'expected_value': [shap_expected_value]})
        shap_expected_value.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'shap', 'shap_expected.csv'),
                                   index=False)
    return shap_values


def _generate_shap_global_values(shap_values, x_df, model_uid, db_schema_name, db_conn, log_to_db):
    """
    Extracts the global shape values for every feature ans saves the outcome as a dataframe locally. Amends the
    dataframe so that it could be used in log_feature_importance_to_mysql().

    :param shap_values: numpy array of shap values
    :param x_df: x_df dataframe
    :param model_uid: model uid
    :param db_schema_name: database schema to log metrics to
    :param log_to_db: Boolean of whether to log scores to the database
    :param db_conn: database connection
    :returns: pandas dataframe
    """
    shap_values = np.abs(shap_values).mean(0)
    df = pd.DataFrame(list(zip(x_df.columns, shap_values)), columns=['feature', 'shap_value'])
    df.sort_values(by=['shap_value'], ascending=False, inplace=True)
    df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'shap', 'shap_global.csv'), index=False)
    df.rename(columns={'shap_value': 'importance_score'}, inplace=True)
    df['model_uid'] = model_uid
    df['importance_metric'] = 'shap'
    df = df[['model_uid', 'feature', 'importance_score', 'importance_metric']]
    if log_to_db:
        log_feature_importance_to_mysql(df, db_schema_name, db_conn)


def _generate_shap_plot(shap_values, x_df, model_uid, plot_type):
    """
    Generates a plot of shap values and saves it locally.

    :param shap_values: numpy array of shap values produced for x_df
    :param x_df: x dataframe
    :param model_uid: model uid
    :param plot_type: the type of plot we want to generate; generally, either dot or bar
    """
    shap.summary_plot(shap_values, x_df, plot_type=plot_type, show=False)
    plt.savefig(os.path.join('modeling', model_uid, 'diagnostics', 'shap', f'shap_values_{plot_type}.png'),
                bbox_inches='tight')
    plt.clf()


def produce_shap_values_and_plots(model, x_df, model_uid, boosting_model, use_kernel, calibrated, db_schema_name,
                                  db_conn, log_to_db):
    """
    Produces SHAP values for x_df and writes associated diagnostics locally.

    :param model: model with predict method
    :param x_df: x dataframe
    :param model_uid: model uid
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param use_kernel: Boolean of whether or not to use Kernel SHAP, which mostly makes sense when we are using a
    CalibratedClassifierCV
    :param calibrated: boolean of whether or not the model is a CalibratedClassifierCV
    :param db_schema_name: database schema to log metrics to
    :param db_conn: database connection
    :param log_to_db: Boolean of whether to log scores to the database
    """
    shap_values = _produce_raw_shap_values(model, model_uid, x_df, calibrated, boosting_model, use_kernel)
    _generate_shap_global_values(shap_values, x_df, model_uid, db_schema_name, db_conn, log_to_db)
    _generate_shap_plot(shap_values, x_df, model_uid, 'dot')
    _generate_shap_plot(shap_values, x_df, model_uid, 'bar')


def pull_out_embedded_feature_scores(x_df, model, model_uid):
    """
    Pulls out feature importance attributes from models and writes the results locally. The passed model must have
    either a coef_ or feature_importances_ attribute.

    :param x_df: cleaned x dataframe
    :param model: model with either a coef_ or feature_importances_ attribute
    :param model_uid: model uid
    """
    if hasattr(model, 'coef_'):
        df = pd.DataFrame({'feature': list(x_df), 'score': model.coef_.reshape(-1)})
        df['score'] = np.exp(df['score'])
        df['score'] = df['score'] / (1 + df['score'])
        df['score'] = (df['score'] - 0.5) / 0.5
        df.rename(columns={'score': 'probability_contribution'}, inplace=True)
    elif hasattr(model, 'feature_importances_'):
        df = pd.DataFrame({'feature': list(x_df), 'importance': model.feature_importances_})
    else:
        raise Exception('model must have either coef_ or feature_importances_ attribute')
    df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'explanation_files', 'embedded_feature_scores.csv'),
              index=False)


def run_permutation_importance(estimator, x_test, y_test, scoring_string, model_uid, db_schema_name, db_conn,
                               log_to_db):
    """
    Produces feature permutation importance scores and saved the results locally.

    :param estimator: estimator object
    :param x_test: x_test
    :param y_test: y_test
    :param scoring_string: scoring metric in the form of a string (e.g. 'neg_log-loss')
    :param model_uid: string name of the model
    :param db_schema_name: database schema to log metrics to
    :param db_conn: database connection
    :param log_to_db: Boolean of whether to log scores to the database
    """
    result = permutation_importance(estimator, x_test, y_test, n_repeats=10, random_state=0, scoring=scoring_string)
    df = pd.DataFrame({
        'permutation_importance_mean': result.importances_mean,
        'permutation_importance_std': result.importances_std,
        'feature': list(x_test)
    })
    df.sort_values(by=['permutation_importance_mean'], ascending=False, inplace=True)
    df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'permutation_importance',
                           'permutation_importance_scores.csv'), index=False)
    df.rename(columns={'permutation_importance_mean': 'importance_score'}, inplace=True)
    df['model_uid'] = model_uid
    df['importance_metric'] = 'permutation_importance'
    df = df[['model_uid', 'feature', 'importance_score', 'importance_metric']]
    if log_to_db:
        log_feature_importance_to_mysql(df, db_schema_name, db_conn)


def _score_drop_col_model(estimator, x_test, y_test, scoring_type, scorer):
    """
    Scores a trained for drop-column feature importance.

    :param estimator: estimator object
    :param x_test: x_test
    :param y_test: y_test
    :param scoring_type: if we want to evaluation class or probability predictions
    :param scorer: scikit-learn scoring callable
    :returns: model's score on the test set
    """
    if scoring_type == 'class':
        predictions = estimator.predict(x_test)
        score = scorer(y_test, predictions)
    elif scoring_type == 'probability':
        predictions = estimator.predict_proba(x_test)
        score = scorer(y_test, predictions[:, 1])
    else:
        raise Exception('scoring_type must either be class or probability')
    return score


def _train_and_score_drop_col_model(feature, estimator, x_train, y_train, x_test, y_test, baseline_score, scoring_type,
                                    scorer):
    """
    Drops specified feature, refits the pipeline to the training data, and determines the differences from the baseline
    model score.

    :param feature: name of the feature to drop
    :param estimator: estimator object
    :param x_train: x_train
    :param y_train: y_train
    :param x_test: x_test
    :param y_test: y_test
    :param baseline_score: the score on the test set using all the columns for training
    :param scoring_type: if we want to evaluation class or probability predictions
    :param scorer: scikit-learn scoring callable
    """
    try:
        x = x_train.drop(feature, axis=1)
        x_test = x_test.drop(feature, axis=1)
        train_pipe = deepcopy(estimator)
        train_pipe.fit(x, y_train)
        feature_score = baseline_score - _score_drop_col_model(train_pipe, x_test, y_test, scoring_type, scorer)
    except:
        feature_score = np.nan
    return {'feature': feature, 'importance': feature_score}


def run_drop_column_importance(pipeline, x_train, y_train, x_test, y_test, scorer, scoring_type, model_uid,
                               db_schema_name, db_conn, log_to_db, higher_is_better):
    """
    Produces drop column feature importance scores and saves the results locally.

    :param pipeline: fitted pipeline
    :param x_train: x_train
    :param y_train: y_train
    :param x_test: x_test
    :param y_test: y_test
    :param scorer: scoring function
    :param scoring_type: either class or probability
    :param model_uid: model uid
    :param db_schema_name: database schema to log metrics to
    :param log_to_db: Boolean of whether to log scores to the database
    :param db_conn: database connection
    :param higher_is_better: whether or not a higher score is better
    """
    pipeline_ = deepcopy(pipeline)
    pipeline_.fit(x_train, y_train)
    baseline_score = _score_drop_col_model(pipeline_, x_test, y_test, scoring_type, scorer)
    drop_col_train_fn = partial(_train_and_score_drop_col_model, estimator=pipeline_, x_train=x_train, y_train=y_train,
                                x_test=x_test, y_test=y_test, baseline_score=baseline_score, scoring_type=scoring_type,
                                scorer=scorer)
    columns = list(x_train)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.map(drop_col_train_fn, columns)
    df = pd.DataFrame.from_records(result)
    df.sort_values(by=['importance'], ascending=higher_is_better, inplace=True)
    df.fillna(0, inplace=True)
    df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'drop_column_importance',
                           'drop_column_importance_scores.csv'), index=False)
    df.rename(columns={'importance': 'importance_score'}, inplace=True)
    df['model_uid'] = model_uid
    df['importance_metric'] = 'drop_column_importance'
    df = df[['model_uid', 'feature', 'importance_score', 'importance_metric']]
    if log_to_db:
        log_feature_importance_to_mysql(df, db_schema_name, db_conn)


def produce_tree_visualization(tree, tree_index, x, y, target_name, feature_names, class_names, model_uid):
    """
    Produces visualization of a decision tree from an ensemble.

    :param tree: tree model
    :param tree_index: index of the tree in the ensemble
    :param x: predictor matrix
    :param y: target series
    :param target_name: name of the target
    :param feature_names: list of feature names
    :param class_names: name of the target classes
    :param model_uid: name of the model
    """
    viz = dtreeviz(tree.estimators_[tree_index],
                   x,
                   y,
                   target_name=target_name,
                   feature_names=feature_names,
                   class_names=class_names
                   )
    viz.save(os.path.join('modeling', model_uid, 'diagnostics', 'trees', f'decision_tree_{tree_index}.svg'))


def _plot_partial_dependence(feature, model, x_df, plot_kind, model_uid):
    """
    Produces a PDP or ICE plot and saves it locally into the pdp directory.

    :param feature: name of the feature
    :param model: fitted model
    :param x_df: x dataframe
    :param plot_kind: "both" for ICE plot of "average" for PDP
    :param model_uid: model uid
    """
    _, ax = plt.subplots(ncols=1, figsize=(9, 4))
    display = plot_partial_dependence(model, x_df, [feature], kind=plot_kind)
    plt.title(feature)
    plt.xlabel(feature)
    plt.savefig(os.path.join('modeling', model_uid, 'diagnostics', 'pdp', f'{feature}_{plot_kind}.png'))
    plt.clf()


def produce_partial_dependence_plots(model, x_df, plot_kind, model_uid):
    """
    Produces a PDP or ICE plot for every column in x_df. x_df is spread across all available CPUs on the machine,
    allowing plots to be created and saved in parallel.

    :param model: fitted model
    :param x_df: x dataframe
    :param plot_kind: "both" for ICE plot of "average" for PDP
    :param model_uid: model uid
    """
    model.fitted_ = True
    pdp_plot_fn = partial(_plot_partial_dependence, model=model, x_df=x_df, plot_kind=plot_kind, model_uid=model_uid)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.map(pdp_plot_fn, list(x_df))


def _produce_ale_plot(feature, x_df, model, model_uid):
    """
    Produces an ALE plot and saves it locally.

    :param feature: name of the feature
    :param x_df: x dataframe
    :param model: fitted model
    :param model_uid: model uid
    """
    ale_effect = ale(X=x_df, model=model, feature=[feature], include_CI=False)
    plt.title(feature)
    plt.xlabel(feature)
    plt.savefig(os.path.join('modeling', model_uid, 'diagnostics', 'ale', f'{feature}_ale.png'))
    plt.clf()


def produce_accumulated_local_effects_plots(x_df, model,  model_uid):
    """
    Produces an ALE plot for every column numereic column in x_df. x_df is spread across all available CPUs on the
    machine, allowing plots to be created and saved in parallel.

    :param x_df: x dataframe
    :param model: fitted model
    feature index as the second item
    :param model_uid: model uid
    """
    x_numeric_df = x_df[find_non_dummied_columns(x_df)]
    ale_plot_fn = partial(_produce_ale_plot, model=model, x_df=x_df, model_uid=model_uid)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.map(ale_plot_fn, list(x_numeric_df))


def run_omnibus_model_explanation(estimator, model_uid, x_test, y_test, x_train, y_train, drop_col_scorer,
                                  drop_col_scorer_string, drop_col_scoring_type, drop_col_higher_is_better,
                                  sample_n, use_shap_kernel, db_schema_name, db_conn, log_to_db):
    """
    Runs a series of model explainability techniques on the model.
    - PDP plots
    - ICE plots
    - ALE plots
    - SHAP values
    - permutation importance
    - drop-column importance

    :param estimator: fitted estimator
    :param x_test: x_test
    :param y_test: y_test
    :param x_train: x_train
    :param y_train: y_train
    :param drop_col_scorer: scikit-learn scoring function
    :param drop_col_scorer_string: scoring metric in the form of a string (e.g. 'neg_log-loss')
    :param drop_col_scoring_type: either class or probability
    :param model_uid: model uid
    :param drop_col_higher_is_better: Boolean of whether or not a higher score is better (e.g. roc auc vs. log loss)
    :param sample_n: number of samples to keep when running interpretability metrics
    :param use_shap_kernel: Boolean of whether or not to use Kernel SHAP, which mostly makes sense when we are using a
    CalibratedClassifierCV
    :param db_schema_name: database schema to log metrics to
    :param db_conn: database connection
    :param log_to_db: Boolean of whether to log scores to the database
    """
    print(f'explaining {model_uid}...')
    make_directories_if_not_exists([
        os.path.join('modeling', model_uid, 'diagnostics', 'shap'),
        os.path.join('modeling', model_uid, 'diagnostics', 'pdp'),
        os.path.join('modeling', model_uid, 'diagnostics', 'ale'),
        os.path.join('modeling', model_uid, 'diagnostics', 'permutation_importance'),
        os.path.join('modeling', model_uid, 'diagnostics', 'drop_column_importance')
    ])

    pipeline_ = deepcopy(estimator)
    model = estimator.named_steps['model']
    x_df = transform_data_with_pipeline(estimator, x_test)
    if sample_n:
        x_train = x_train.head(sample_n)
        y_train = y_train.head(sample_n)
        x_test = x_test.head(sample_n)
        y_test = y_test.head(sample_n)
        x_df = x_df.head(sample_n)

    boosting_model = determine_if_name_in_object('boost', model)
    calibrated_model = determine_if_name_in_object('calibrated', model)
    produce_shap_values_and_plots(model, x_df, model_uid, boosting_model, use_shap_kernel, calibrated_model,
                                  db_schema_name, db_conn, log_to_db)
    produce_partial_dependence_plots(model, x_df, 'average', model_uid)
    produce_partial_dependence_plots(model, x_df, 'both', model_uid)
    produce_accumulated_local_effects_plots(x_df, model, model_uid)
    run_permutation_importance(model, x_df, y_test, drop_col_scorer_string, model_uid, db_schema_name, db_conn,
                               log_to_db)
    run_drop_column_importance(pipeline_, x_train, y_train, x_test, y_test, drop_col_scorer, drop_col_scoring_type,
                               model_uid, db_schema_name, db_conn, log_to_db, drop_col_higher_is_better)
