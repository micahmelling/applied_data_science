import pandas as pd
import numpy as np
import os
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from copy import deepcopy
from statistics import mean
from sklearn.calibration import calibration_curve
from sklearn.metrics import plot_roc_curve, roc_curve, confusion_matrix
from mlxtend.evaluate import lift_score
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
    fig.suptitle(f'Calibration Plot')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability in Each Bin')
    plt.legend()
    plt.savefig(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{model_name}_{bin_strategy}_calibration_plot.png'))
    plt.clf()
    calibration_df = pd.DataFrame({'prob_true': prob_true, 'prob_pred': prob_pred})
    calibration_df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY,
                                       f'{model_name}_{bin_strategy}_calibration_summary.csv'), index=False)


def produce_roc_curve_plot(pipeline, x_test, y_test, model_name):
    """
    Produces a ROC curve plot and saves the result locally.

    :param pipeline: scikit-learn modeling pipeline
    :param x_test: x_test dataframe
    :param y_test: y_test series
    :param model_name: string name of the model
    """
    plot_roc_curve(pipeline, x_test, y_test)
    plt.savefig(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'roc_curve.png'))
    plt.clf()


def find_optimal_class_cutoff(y_test, predictions, model_name):
    """
    Finds the optimal class cutoff based on the ROC curve and saves the result locally.

    :param y_test: y_test series
    :param predictions: probability predictions for the positive class
    :param model_name: string name of the model
    """
    fpr, tpr, threshold = roc_curve(y_test, predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    best_tpr = tpr[optimal_idx]
    best_fpr = fpr[optimal_idx]
    df = pd.DataFrame({'optimal_threshold': [optimal_threshold], 'best_tpr': [best_tpr], 'best_fpr': [best_fpr]})
    df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'optimal_class_cutoff.csv'), index=False)
    return


def plot_confusion_matrix(y_test, class_predictions, model_name):
    """
    Plots a confusion matrix ans saves it locally.

    :param y_test: y_test series
    :param class_predictions: class predictions series
    :param model_name: string name of the model
    """
    data_cm = confusion_matrix(y_test, class_predictions)
    tn, fp, fn, tp = data_cm.ravel()
    fpr = round((fp / (fp + tn)) * 100, 2)
    fnr = round((fn / (fn + tp)) * 100, 2)
    df_cm = pd.DataFrame(data_cm, columns=np.unique(y_test), index=np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, cbar=False, fmt='g')
    plt.suptitle('Confusion Matrix')
    plt.title(f'FPR: {fpr}%    FNR: {fnr}%')
    plt.savefig(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'confusion_matrix.png'))
    plt.clf()


def get_bootstrap_estimate(y_test, prediction_series, scoring_metric, metric_name, model_name, samples=30):
    """
    Produces a bootstrap estimate for a scoring metric. y_test and it's associated predictions are sampled with
    replacement. The samples argument dictates how many times to perform each sampling. Each sample is of equal length,
    determined by len(y_test) / samples. Summary statistics and a density plot of the distribution are saved locally.

    :param y_test: y_test series
    :param prediction_series: relevant prediction series
    :param scoring_metric: scikit-learn scoring metric callable (e.g. roc_auc_score)
    :param metric_name: string name of the metric
    :param model_name: string name of the model
    :param samples: number of samples to take; default is 30
    """
    y_test = y_test.reset_index(drop=True)
    prediction_series = prediction_series.reset_index(drop=True)
    df = pd.concat([y_test, prediction_series], axis=1)
    approx_sample_size = int(len(df) / samples)
    metrics_df = pd.DataFrame()

    for sample in range(samples):
        temp_df = df.sample(n=approx_sample_size)
        score = scoring_metric(temp_df.iloc[:, 0], temp_df.iloc[:, 1])
        temp_df = pd.DataFrame({'score': [score]})
        metrics_df = metrics_df.append(temp_df)
    described_df = metrics_df.describe()
    described_df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{metric_name}_sample_scores.csv'))

    sns.kdeplot(metrics_df['score'], shade=True, color='b', legend=False)
    plt.xlabel(metric_name)
    plt.ylabel('density')
    plt.title(f'{metric_name} distribution')
    plt.savefig(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{metric_name}_density_plot.png'))
    plt.clf()


def calculate_class_lift(y_test, class_predictions, model_name):
    """
    Calculates the lift of a model, based on predicted class labels.

    :param y_test: y_test series
    :param class_predictions: class predictions series
    :param model_name: string name of the model
    """
    lift = lift_score(y_test, class_predictions)
    pd.DataFrame({'lift': [lift]}).to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'class_lift.csv'))


def calculate_probability_lift(y_test, probability_predictions, model_name, bins=10):
    """
    Calculates the lift provided by the probability estimates.

    :param y_test: y_test series
    :param probability_predictions: positive probability predictions series
    :param model_name: string name of the model
    :param bins: number of bins to segment the probability predictions
    :param model_name: string name of the model
    """
    y_test = y_test.reset_index(drop=True)
    prediction_series = probability_predictions.reset_index(drop=True)
    df = pd.concat([y_test, prediction_series], axis=1)
    columns = list(df)
    class_col = columns[0]
    proba_col = columns[1]
    base_rate = df[class_col].mean()

    df['1_prob_bin'] = pd.qcut(df[proba_col], q=bins, labels=list(range(1, 11)))
    grouped_df = df.groupby('1_prob_bin').agg({proba_col: 'mean', class_col: 'mean'})
    grouped_df.reset_index(inplace=True)
    grouped_df['1_prob_diff'] = grouped_df[proba_col] - grouped_df[class_col]
    grouped_df['base_rate_diff'] = base_rate - grouped_df[class_col]

    prob_diff = grouped_df['1_prob_diff'].abs().sum()
    base_rate_diff = grouped_df['base_rate_diff'].abs().sum()
    lift = base_rate_diff / prob_diff
    pd.DataFrame({'lift': [lift]}).to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'proba_lift.csv'))


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
                                 calibration_bins, calibrated=False):
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
    :param calibrated: boolean of whether or not the pipeline has a CalibratedClassifierCV; default is False
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
