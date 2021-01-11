import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from copy import deepcopy
from statistics import mean
from sklearn.calibration import calibration_curve
from tqdm import tqdm

from modeling.config import DIAGNOSTICS_DIRECTORY, SCHEMA_NAME
from data.db import log_feature_importance_to_mysql, log_model_scores_to_mysql


def produce_predictions(pipeline, model_name, x_test, y_test, class_cutoff):
    """
    Produces a dataframe consisting of the probability and class predictions from the model on the test set, along
    with the y_test and x_test values. The dataframe is saved locally.

    :param pipeline: scikit-learn modeling pipeline
    :param model_name: name of the model
    :param x_test: x_test dataframe
    :param y_test: y_test series
    :param class_cutoff: the probability cutoff to separate classes
    :returns: pandas dataframe
    """
    df = pd.concat(
        [
            pd.DataFrame(pipeline.predict_proba(x_test), columns=['0_prob', '1_prob']),
            y_test.reset_index(drop=True),
            x_test.reset_index(drop=True)
        ],
        axis=1)
    df['predicted_class'] = np.where(df['1_prob'] >= class_cutoff, 1, 0)
    df = df[['predicted_class'] + [col for col in df.columns if col != 'predicted_class']]
    df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_predictions.csv'), index=False)
    return df


def _evaluate_model(df, target, predictions, scorer, metric_name):
    """
    Applies a scorer function to evaluate predictions against the ground-truth labels.

    :param df: pandas dataframe containing the predictions and the actuals
    :param target: name of the target column in df
    :param predictions: name of the column with the predictions
    :param scorer: scoring function to evaluate the predictions
    :param metric_name: name of the metric we are using to score our model

    :returns: pandas dataframe
    """
    score = scorer(df[target], df[predictions])
    df = pd.DataFrame({metric_name: [score]})
    return df


def run_evaluation_metrics(df, target, model_name, evaluation_list):
    """
    Runs a series of evaluations metrics on a model's predictions and writes the results locally.

    :param df: pandas dataframe containing the predictions and the actuals
    :param target: name of the target column
    :param model_name: name of the model
    :param evaluation_list: list of tuples, which each tuple having the ordering of: the column with the predictions,
    the scoring function callable, and the name of the metric
    """
    main_df = pd.DataFrame()
    for metric_config in evaluation_list:
        temp_df = _evaluate_model(df, target, metric_config[0], metric_config[1], metric_config[2])
        main_df = pd.concat([main_df, temp_df], axis=1)
    main_df = main_df.T
    main_df.reset_index(inplace=True)
    main_df.columns = ['scoring_metric', 'holdout_score']
    main_df['model_uid'] = model_name
    main_df['holdout_type'] = 'test'
    log_model_scores_to_mysql(main_df, SCHEMA_NAME)
    main_df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_evaluation.csv'), index=False)


def plot_calibration_curve(y_test, predictions, n_bins, bin_strategy, model_name):
    """
    Produces a calibration plot and saves it locally.

    :param y_test: y_test series
    :param predictions: predictions series
    :param n_bins: number of bins for the predictions
    :param bin_strategy: uniform - all bins have the same width; quantile - bins have the same number of observations
    :param model_name: name of the model
    """
    prob_true, prob_pred = calibration_curve(y_test, predictions, n_bins=n_bins, strategy=bin_strategy)
    fig, ax = plt.subplots()
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='model')
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle('Calibration Plot')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability in Each Bin')
    plt.legend()
    plt.savefig(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_{bin_strategy}_calibration_plot.png'))
    plt.clf()
    calibration_df = pd.DataFrame({'prob_true': prob_true, 'prob_pred': prob_pred})
    calibration_df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY,
                                       f'{model_name}_{bin_strategy}_calibration_summary.csv'), index=False)


def _produce_raw_shap_values(model, model_name, x_test, calibrated):
    """
    Produces the raw shap values for every observation in the test set. A dataframe of the shap values is saved locally
    as a csv. The shap expected value is extracted and save locally in a csv.

    :param model: fitted model
    :param model_name: name of the model
    :param x_test: x_test
    :param calibrated: boolean of whether or not the model is a CalibratedClassifierCV; the default is False
    :returns: numpy array
    """
    if calibrated:
        shap_values_list = []
        shap_expected_list = []
        for calibrated_classifier in tqdm(model.calibrated_classifiers_):
            explainer = shap.TreeExplainer(calibrated_classifier.base_estimator)
            shap_values = explainer.shap_values(x_test)[1]
            shap_expected_value = explainer.expected_value[1]
            shap_values_list.append(shap_values)
            shap_expected_list.append(shap_expected_value)
        shap_values = np.array(shap_values_list).sum(axis=0) / len(shap_values_list)
        shap_expected_value = mean(shap_expected_list)
        shap_df = pd.DataFrame(shap_values, columns=list(x_test))
        shap_df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_shap_values.csv'), index=False)
        shap_expected_value = pd.DataFrame({'expected_value': [shap_expected_value]})
        shap_expected_value.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_shap_expected.csv'),
                                   index=False)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)[1]
        shap_df = pd.DataFrame(shap_values, columns=list(x_test))
        shap_df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_shap_values.csv'), index=False)
        shap_expected_value = pd.DataFrame({'expected_value': [explainer.expected_value[1]]})
        shap_expected_value.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_shap_expected.csv'),
                                   index=False)
    return shap_values


def _generate_shap_global_values(shap_values, x_test, model_name):
    """
    Extracts the global shape values for every feature ans saves the outcome as a dataframe locally. Amends the
    dataframe so that it could be used in log_feature_importance_to_mysql().

    :param shap_values: numpy array of shap values
    :param x_test: x_test dataframe
    :param model_name: string name of the model
    :returns: pandas dataframe
    """
    shap_values = np.abs(shap_values).mean(0)
    df = pd.DataFrame(list(zip(x_test.columns, shap_values)), columns=['feature', 'shap_value'])
    df.sort_values(by=['shap_value'], ascending=False, inplace=True)
    df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_shap_global.csv'), index=False)
    df.rename(columns={'shap_value': 'importance_score'}, inplace=True)
    df['model_uid'] = model_name
    df['importance_metric'] = 'shap'
    return df


def _generate_shap_plot(shap_values, x_test, model_name, plot_type):
    """
    Generates a plot of shap values and saves it locally.

    :param shap_values: numpy array of shap values produced for x_test
    :param x_test: x_test dataframe
    :param model_name: string name of the model
    :param plot_type: the type of plot we want to generate; generally, either dot or bar
    """
    shap.summary_plot(shap_values, x_test, plot_type=plot_type, show=False)
    plt.savefig(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_shap_values_{plot_type}.png'),
                bbox_inches='tight')
    plt.clf()


def _prepare_data_for_shap(pipeline, x_test, n_obs=10_000):
    """
    Prepares the model and x_test dataframe for extracting SHAP values. This involves applying the preprocessing steps
    in the pipeline and converting the output into a dataframe with the appropriate columns. Likewise, this process
    involves plucking out the model, as the SHAP functions only take models and not pipelines.

    :param pipeline: scikit-learn pipeline with preprocessing steps and model
    :param x_test: x_test dataframe
    :param n_obs: the number of observations to keep in x_test because calculating SHAP values can be quite
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

    # Isolate the model in the pipeline
    model = pipeline.named_steps['model']

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

    try:
        x_test = x_test.sample(n=n_obs)
    except ValueError:
        x_test = x_test.sample(frac=0.5)

    return model, x_test


def produce_shap_values(model, x_test, model_name, calibrated=False):
    """
    Produces SHAP values for x_test and writes associated diagnostics locally.

    :param model: model with predict method
    :param x_test: x_test
    :param model_name: name of the model
    :param calibrated: boolean of whether or not the model is a CalibratedClassifierCV; the default is False
    """
    shap_values = _produce_raw_shap_values(model, model_name, x_test, calibrated)
    global_shap_df = _generate_shap_global_values(shap_values, x_test, model_name)
    log_feature_importance_to_mysql(global_shap_df, SCHEMA_NAME)
    _generate_shap_plot(shap_values, x_test, model_name, 'dot')
    _generate_shap_plot(shap_values, x_test, model_name, 'bar')


def run_omnibus_model_evaluation(pipeline, model_name, x_test, y_test, class_cutoff, target, evaluation_list,
                                 calibration_bins, calibrated=True):
    """
    Runs a series of functions to evaluate a model's performance.

    :param pipeline: scikit-learn pipeline
    :param model_name: name of the model
    :param x_test: x_test dataframe
    :param y_test: y_test series
    :param class_cutoff: the probability cutoff to separate classes
    :param target: name of the target column
    :param evaluation_list: list of tuples, which each tuple having the ordering of: the column with the predictions,
    the scoring function callable, and the name of the metric
    :param calibration_bins: number of bins in the calibration plot
    :param calibrated: boolean of whether or not the pipeline has a CalibratedClassifierCV
    """
    print(f'evaluating {model_name}...')
    predictions_df = produce_predictions(pipeline, model_name, x_test, y_test, class_cutoff)
    run_evaluation_metrics(predictions_df, target, model_name, evaluation_list)
    plot_calibration_curve(y_test, predictions_df['1_prob'], calibration_bins, 'uniform', model_name)
    plot_calibration_curve(y_test, predictions_df['1_prob'], calibration_bins, 'quantile', model_name)
    model, x_test = _prepare_data_for_shap(pipeline, x_test)
    try:
        produce_shap_values(model, x_test, model_name, calibrated=calibrated)
    except Exception as e:
        print(e)
        print(f'unable to run shap values for {model_name}')
