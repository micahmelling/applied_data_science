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

from data.db import log_feature_importance_to_mysql
from helpers.model_helpers import make_directories_if_not_exists, find_non_dummied_columns
from modeling.config import DIAGNOSTICS_DIRECTORY, SCHEMA_NAME


plt.switch_backend('Agg')


def transform_data_with_pipeline(pipeline, x_test):
    """
    Prepares the model and x_test dataframe for extracting feature importance values. This involves applying the
    preprocessing stepsin the pipeline and converting the output into a dataframe with the appropriate columns.
    Likewise, this processinvolves plucking out the model.

    :param pipeline: scikit-learn pipeline with preprocessing steps and model
    :param x_test: x_test dataframe
    computationally expensive; default is 10,000. If n_obs is greater than the total number of observations, then
    50% of the data will be sampled.
    :returns: model with predict method, transformed x_test dataframe
    """
    # Extract the names of the features from the dict vectorizers
    num_dict_vect = pipeline.named_steps['preprocessor'].named_transformers_.get('numeric_transformer').named_steps[
        'dict_vectorizer']
    cat_dict_vect = pipeline.named_steps['preprocessor'].named_transformers_.get('categorical_transformer').named_steps[
        'dict_vectorizer']
    num_features = num_dict_vect.feature_names_
    cat_features = cat_dict_vect.feature_names_

    # Get the boolean masks for the variance threshold and feature selector steps
    variance_threshold_support = pipeline.named_steps['variance_thresholder'].get_support()
    feature_selector_support = pipeline.named_steps['feature_selector'].get_support()

    # Create a dataframe of column names
    cols_df = pd.DataFrame({'cols': num_features + cat_features,
                            'variance_threshold_support': variance_threshold_support})
    cols = cols_df['cols'].tolist()

    # Remove the model
    pipeline.steps.pop(len(pipeline) - 1)

    # Remove the feature selector
    pipeline.steps.pop(len(pipeline) - 1)

    # Remove the variance threshold
    pipeline.steps.pop(len(pipeline) - 1)

    # Transform the data using the remaining pipeline steps, cast to a dataframe, and assign the column names
    x_test = pipeline.transform(x_test)
    x_test = pd.DataFrame(x_test)
    x_test.columns = cols

    # Remove the columns taken out by the variance threshold
    remove_df = deepcopy(cols_df)
    remove_df = remove_df.loc[remove_df['variance_threshold_support'] == False]
    remove_cols = remove_df['cols'].tolist()
    x_test.drop(remove_cols, 1, inplace=True)

    # Create a dataframe for the feature selector step
    cols_df = cols_df.loc[cols_df['variance_threshold_support'] == True]
    cols_df.reset_index(inplace=True, drop=True)
    feature_selector_df = pd.DataFrame({'feature_selector_support': feature_selector_support})
    cols_df = pd.concat([cols_df, feature_selector_df], axis=1)

    # Remove the columns taken out by the variance threshold
    remove_df = cols_df.loc[cols_df['feature_selector_support'] == False]
    remove_cols = remove_df['cols'].tolist()
    x_test.drop(remove_cols, 1, inplace=True)
    return x_test


def _run_shap_explainer(x_df, explainer, boosting_model, use_kernel):
    """
    Runs the SHAP explainer on a dataframe.

    :param x_df: x dataframe
    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param use_kernel: Boolean of whether or not to use Kernel SHAP, which mostly makes sense when we are using a
    CalibratedClassifierCV
    """
    if boosting_model:
        if use_kernel:
            return explainer.shap_values(x_df, nsamples=500, check_additivity=False)
        else:
            return explainer.shap_values(x_df, check_additivity=False)
    else:
        if use_kernel:
            return explainer.shap_values(x_df, nsamples=500, check_additivity=False)[1]
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
    pool = mp.Pool(processes=mp.cpu_count())
    shap_fn = partial(_run_shap_explainer, explainer=explainer, boosting_model=boosting_model, use_kernel=use_kernel)
    result = pool.map(shap_fn, array_split)
    pool.close()
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
        shap_df.to_csv(os.path.join(model_uid, 'diagnostics', 'shap', 'shap_values.csv'), index=False)
        shap_expected_value = pd.DataFrame({'expected_value': [shap_expected_value]})
        shap_expected_value.to_csv(os.path.join(model_uid, 'diagnostics', 'shap', 'shap_expected.csv'), index=False)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = _run_parallel_shap_explainer(x_df, explainer, boosting_model, False)
        shap_df = pd.DataFrame(shap_values, columns=list(x_df))
        shap_df.to_csv(os.path.join(model_uid, 'diagnostics', 'shap', 'shap_values.csv'), index=False)
        shap_expected_value = _get_shap_expected_value(explainer, boosting_model)
        shap_expected_value = pd.DataFrame({'expected_value': [shap_expected_value]})
        shap_expected_value.to_csv(os.path.join(model_uid, 'diagnostics', 'shap', 'shap_expected.csv'), index=False)
    return shap_values


def _generate_shap_global_values(shap_values, x_df, model_uid):
    """
    Extracts the global shape values for every feature ans saves the outcome as a dataframe locally. Amends the
    dataframe so that it could be used in log_feature_importance_to_mysql().

    :param shap_values: numpy array of shap values
    :param x_df: x_df dataframe
    :param model_uid: model uid
    :returns: pandas dataframe
    """
    shap_values = np.abs(shap_values).mean(0)
    df = pd.DataFrame(list(zip(x_df.columns, shap_values)), columns=['feature', 'shap_value'])
    df.sort_values(by=['shap_value'], ascending=False, inplace=True)
    df.to_csv(os.path.join(model_uid, 'diagnostics', 'shap', 'shap_global.csv'), index=False)
    df.rename(columns={'shap_value': 'importance_score'}, inplace=True)
    df['model_uid'] = model_uid
    df['importance_metric'] = 'shap'
    df = df[['model_uid', 'feature', 'importance_score', 'importance_metric']]
    log_feature_importance_to_mysql(df, SCHEMA_NAME)


def _generate_shap_plot(shap_values, x_df, model_uid, plot_type):
    """
    Generates a plot of shap values and saves it locally.

    :param shap_values: numpy array of shap values produced for x_df
    :param x_df: x dataframe
    :param model_uid: model uid
    :param plot_type: the type of plot we want to generate; generally, either dot or bar
    """
    shap.summary_plot(shap_values, x_df, plot_type=plot_type, show=False)
    plt.savefig(os.path.join(model_uid, 'diagnostics', 'shap', f'shap_values_{plot_type}.png'),
                bbox_inches='tight')
    plt.clf()


def produce_shap_values_and_plots(model, x_df, model_uid, boosting_model, use_kernel, calibrated):
    """
    Produces SHAP values for x_df and writes associated diagnostics locally.

    :param model: model with predict method
    :param x_df: x dataframe
    :param model_uid: model uid
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param use_kernel: Boolean of whether or not to use Kernel SHAP, which mostly makes sense when we are using a
    CalibratedClassifierCV
    :param calibrated: boolean of whether or not the model is a CalibratedClassifierCV
    """
    make_directories_if_not_exists([os.path.join(model_uid, 'diagnostics', 'shap')])
    shap_values = _produce_raw_shap_values(model, model_uid, x_df, calibrated, boosting_model, use_kernel)
    _generate_shap_global_values(shap_values, x_df, model_uid)
    _generate_shap_plot(shap_values, x_df, model_uid, 'dot')
    _generate_shap_plot(shap_values, x_df, model_uid, 'bar')


def pull_out_embedded_feature_scores(cleaned_x_test, model, model_name):
    """
    Pulls out feature importance attributes from models and writes the results locally. The passed model must have
    either a coef_ or feature_importances_ attribute.

    :param cleaned_x_test: cleaned x_test dataframe generated by _isolate_model_and_preprocess_data
    :param model: model with either a coef_ or feature_importances_ attribute
    :param model_name: name of the model
    """
    if hasattr(model, 'coef_'):
        df = pd.DataFrame({'feature': list(cleaned_x_test), 'score': model.coef_.reshape(-1)})
        df['score'] = np.exp(df['score'])
        df['score'] = df['score'] / (1 + df['score'])
        df['score'] = (df['score'] - 0.5) / 0.5
        df.rename(columns={'score': 'probability_contribution'}, inplace=True)
    elif hasattr(model, 'feature_importances_'):
        df = pd.DataFrame({'feature': list(cleaned_x_test), 'importance': model.feature_importances_})
    else:
        raise Exception('model must have either coef_ or feature_importances_ attribute')
    df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'embedded_feature_scores.csv'), index=False)


def run_permutation_importance(pipeline, x_test, y_test, scoring_string, model_uid):
    """
    Produces feature permutation importance scores and saved the results locally.

    :param pipeline: fitted pipeline
    :param x_test: x_test
    :param y_test: y_test
    :param scoring_string: scoring metric in the form of a string (e.g. 'neg_log-loss')
    :param model_uid: string name of the model
    """
    result = permutation_importance(pipeline, x_test, y_test, n_repeats=10, random_state=0, scoring=scoring_string)
    df = pd.DataFrame({
        'permutation_importance_mean': result.importances_mean,
        'permutation_importance_std': result.importances_std,
        'feature': list(x_test)
    })
    df.sort_values(by=['permutation_importance_mean'], ascending=False, inplace=True)
    df.to_csv(os.path.join(model_uid, DIAGNOSTICS_DIRECTORY, 'permutation_importance_scores.csv'), index=False)
    df.rename(columns={'permutation_importance_mean': 'importance_score'}, inplace=True)
    df['model_uid'] = model_uid
    df['importance_metric'] = 'permutation_importance'
    df = df[['model_uid', 'feature', 'importance_score', 'importance_metric']]
    log_feature_importance_to_mysql(df, SCHEMA_NAME)


def _score_drop_col_model(pipe, x_test, y_test, scoring_type, scorer):
    """
    Scores a trained for drop-column feature importance.

    :param pipe: scikit-learn pipeline
    :param x_test: x_test
    :param y_test: y_test
    :param scoring_type: if we want to evaluation class or probability predictions
    :param scorer: scikit-learn scoring callable
    :returns: model's score on the test set
    """
    if scoring_type == 'class':
        predictions = pipe.predict(x_test)
        score = scorer(y_test, predictions)
    elif scoring_type == 'probability':
        predictions = pipe.predict_proba(x_test)
        score = scorer(y_test, predictions[:, 1])
    else:
        raise Exception('scoring_type must either be class or probability')
    return score


def _train_and_score_drop_col_model(feature, pipeline, x_train, y_train, x_test, y_test, baseline_score, scoring_type,
                                    scorer):
    """
    Drops specified feature, refits the pipeline to the training data, and determines the differences from the baseline
    model score.

    :param feature: name of the feature to drop
    :param pipeline: fitted scikit-learn pipeline
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
        train_pipe = deepcopy(pipeline)
        train_pipe.fit(x, y_train)
        feature_score = baseline_score - _score_drop_col_model(train_pipe, x_test, y_test, scoring_type, scorer)
    except:
        feature_score = np.nan
    return {'feature': feature, 'importance': feature_score}


def run_drop_column_importance(pipeline, x_train, y_train, x_test, y_test, scorer, scoring_type, model_uid,
                               higher_is_better=True):
    """
    Produces drop column feature importance scores and saves the results locally.

    :param pipeline: fitted pipeline
    :param x_train: x_train
    :param y_train: y_train
    :param x_test: x_test
    :param y_test: y_test
    :param scorer: scoring function
    :param scoring_type: either class or probability
    :param model_uid: string name of the model
    :param higher_is_better: whether or not a higher score is better
    """
    pipeline_ = deepcopy(pipeline)
    pipeline_.fit(x_train, y_train)
    baseline_score = _score_drop_col_model(pipeline_, x_test, y_test, scoring_type, scorer)
    drop_col_train_fn = partial(_train_and_score_drop_col_model, pipeline=pipeline_, x_train=x_train, y_train=y_train,
                                x_test=x_test, y_test=y_test, baseline_score=baseline_score, scoring_type=scoring_type,
                                scorer=scorer)
    columns = list(x_train)
    pool = mp.Pool(processes=mp.cpu_count())
    result = pool.map(drop_col_train_fn, columns)
    df = pd.DataFrame.from_records(result)
    df.sort_values(by=['importance'], ascending=higher_is_better, inplace=True)
    df.fillna(0, inplace=True)
    df.to_csv(os.path.join(model_uid, DIAGNOSTICS_DIRECTORY, 'drop_column_importance_scores.csv'), index=False)
    df.rename(columns={'importance': 'importance_score'}, inplace=True)
    df['model_uid'] = model_uid
    df['importance_metric'] = 'drop_column_importance'
    df = df[['model_uid', 'feature', 'importance_score', 'importance_metric']]
    log_feature_importance_to_mysql(df, SCHEMA_NAME)


def produce_tree_visualization(tree, tree_index, x, y, target_name, feature_names, class_names, model_name):
    """
    Produces visualization of a decision tree from an ensemble.

    :param tree: tree model
    :param tree_index: index of the tree in the ensemble
    :param x: predictor matrix
    :param y: target series
    :param target_name: name of the target
    :param feature_names: list of feature names
    :param class_names: name of the target classes
    :param model_name: name of the model
    """
    viz = dtreeviz(tree.estimators_[tree_index],
                   x,
                   y,
                   target_name=target_name,
                   feature_names=feature_names,
                   class_names=class_names
                   )
    viz.save(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'decision_tree_{tree_index}.svg'))


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
    plt.savefig(os.path.join(model_uid, 'diagnostics', 'pdp', f'{feature}_{plot_kind}.png'))
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
    make_directories_if_not_exists([os.path.join(model_uid, 'diagnostics', 'pdp')])
    pool = mp.Pool(processes=mp.cpu_count())
    pdp_plot_fn = partial(_plot_partial_dependence, model=model, x_df=x_df, plot_kind=plot_kind, model_uid=model_uid)
    result = pool.map(pdp_plot_fn, list(x_df))
    pool.close()


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
    plt.savefig(os.path.join(model_uid, 'diagnostics', 'ale', f'{feature}_ale.png'))
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
    make_directories_if_not_exists([os.path.join(model_uid, 'diagnostics', 'ale')])
    pool = mp.Pool(processes=mp.cpu_count())
    ale_plot_fn = partial(_produce_ale_plot, model=model, x_df=x_df, model_uid=model_uid)
    result = pool.map(ale_plot_fn, list(x_numeric_df))
    pool.close()


def run_omnibus_model_explanation(pipeline, x_test, y_test, x_train, y_train, scorer, scorer_string, scoring_type,
                                  model_uid, higher_is_better, calibrated_model, sample_n, use_kernel):
    """
    Runs a series of model explainability techniques on the model.
    - PDP plots
    - ICE plots
    - ALE plots
    - SHAP values
    - permutation importance
    - drop-column importance

    :param pipeline: scikit-learn pipeline
    :param x_test: x_test
    :param y_test: y_test
    :param x_train: x_train
    :param y_train: y_train
    :param scorer: scikit-learn scoring function
    :param scorer_string: scoring metric in the form of a string (e.g. 'neg_log-loss')
    :param scoring_type: either class or probability
    :param model_uid: model uid
    :param higher_is_better: Boolean of whether or not a higher score is better (e.g. roc auc vs. log loss)
    :param calibrated_model: Boolean of whether or not the model is a CalibratedClassifierCV
    :param sample_n: number of samples to keep when running interpretability metrics
    :param use_kernel: Boolean of whether or not to use Kernel SHAP, which mostly makes sense when we are using a
    CalibratedClassifierCV
    """
    print(f'explaining {model_uid}...')

    pipeline_ = deepcopy(pipeline)
    model = pipeline.named_steps['model']
    x_df = transform_data_with_pipeline(pipeline, x_test)
    x_train = x_train.head(sample_n)
    y_train = y_train.head(sample_n)
    x_test = x_test.head(sample_n)
    y_test = y_test.head(sample_n)
    x_df = x_df.head(sample_n)

    model_type = str((type(model))).lower()
    if 'boost' in model_type:
        boosting_model = True
    else:
        boosting_model = False
    produce_shap_values_and_plots(model, x_df, model_uid, boosting_model, calibrated_model, use_kernel)

    produce_partial_dependence_plots(model, x_df, 'average', model_uid)
    produce_partial_dependence_plots(model, x_df, 'both', model_uid)
    produce_accumulated_local_effects_plots(x_df, model, model_uid)
    run_permutation_importance(model, x_df, y_test, scorer_string, model_uid)
    run_drop_column_importance(pipeline_, x_train, y_train, x_test, y_test, scorer, scoring_type, model_uid,
                               higher_is_better)
