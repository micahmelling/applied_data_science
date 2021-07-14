import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scikitplot as skplt
import itertools
import multiprocessing as mp

from sklearn.calibration import calibration_curve
from sklearn.metrics import plot_roc_curve, roc_curve, confusion_matrix
from mlxtend.evaluate import lift_score
from mlxtend.evaluate import mcnemar, mcnemar_table, cochrans_q, bias_variance_decomp
from mlxtend.plotting import checkerboard_plot
from scipy.stats import ks_2samp

from modeling.config import DIAGNOSTICS_DIRECTORY, SCHEMA_NAME
from data.db import log_model_scores_to_mysql


def produce_predictions(pipeline, model_name, x_test, y_test, class_cutoff):
    """
    Produces a dataframe consisting of the probability and class predictions from the model on the test set along
    with the y_test. The dataframe is saved locally.

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
            y_test.reset_index(drop=True)
        ],
        axis=1)
    df['predicted_class'] = np.where(df['1_prob'] >= class_cutoff, 1, 0)
    df = df[['predicted_class'] + [col for col in df.columns if col != 'predicted_class']]
    df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'predictions_vs_actuals.csv'), index=False)
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
    main_df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'evaluation_scores.csv'), index=False)


def plot_calibration_curve(y_test, predictions, n_bins, bin_strategy, model_name):
    """
    Produces a calibration plot and saves it locally. The raw data behind the plot is also written locally.

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
    plt.savefig(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'{bin_strategy}_{n_bins}_calibration_plot.png'))
    plt.clf()
    calibration_df = pd.DataFrame({'prob_true': prob_true, 'prob_pred': prob_pred})
    calibration_df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY,
                                       f'{bin_strategy}_{n_bins}_calibration_summary.csv'), index=False)


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


def find_optimal_class_cutoff(y_test, probability_predictions, model_name):
    """
    Finds the optimal class cutoff based on the ROC curve and saves the result locally.

    :param y_test: y_test series
    :param probability_predictions: probability predictions for the positive class
    :param model_name: string name of the model
    :return: float that represents the optimal threshold
    """
    fpr, tpr, threshold = roc_curve(y_test, probability_predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    best_tpr = tpr[optimal_idx]
    best_fpr = fpr[optimal_idx]
    df = pd.DataFrame({'optimal_threshold': [optimal_threshold], 'best_tpr': [best_tpr], 'best_fpr': [best_fpr]})
    df.to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'optimal_class_cutoff.csv'), index=False)
    return optimal_threshold


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


def calculate_probability_lift(y_test, probability_predictions, model_name, n_bins=10):
    """
    Calculates the lift provided by the probability estimates. Lift is determined by how much improvement is experienced
    by using the predicted probabilities over assuming that each observation has the same probability of being in the
    positive class (i.e. applying the overall rate of occurrence of the positive class to all observations).

    This process takes the following steps:
    - find the overall rate of occurrence of the positive class
    - cut the probability estimates into n_bins
    - for each bin, calculate:
       - the average predicted probability
       - the actual probability
    -  for each bin, calculate
       - the difference between the average predicted probability and the true probability
       - the difference between the overall rate of occurrence and the true probability
    - take the sum of the absolute value for each the differences calculated in the previous step
    - take the ratio of the two sums, with the base rate sum as the numerator

    Values above 1 indicate the predicted probabilities have lift over simply assuming each observation has the same
    probability.

    :param y_test: y_test series
    :param probability_predictions: positive probability predictions series
    :param model_name: string name of the model
    :param n_bins: number of bins to segment the probability predictions
    """
    y_test = y_test.reset_index(drop=True)
    prediction_series = probability_predictions.reset_index(drop=True)
    df = pd.concat([y_test, prediction_series], axis=1)
    columns = list(df)
    class_col = columns[0]
    proba_col = columns[1]
    base_rate = df[class_col].mean()

    df['1_prob_bin'] = pd.qcut(df[proba_col], q=n_bins, labels=list(range(1, 11)))
    grouped_df = df.groupby('1_prob_bin').agg({proba_col: 'mean', class_col: 'mean'})
    grouped_df.reset_index(inplace=True)
    grouped_df['1_prob_diff'] = grouped_df[proba_col] - grouped_df[class_col]
    grouped_df['base_rate_diff'] = base_rate - grouped_df[class_col]

    prob_diff = grouped_df['1_prob_diff'].abs().sum()
    base_rate_diff = grouped_df['base_rate_diff'].abs().sum()
    lift = base_rate_diff / prob_diff
    pd.DataFrame({'lift': [lift]}).to_csv(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'proba_lift.csv'))


def plot_cumulative_gains_chart(y_test, probability_predictions, model_name):
    """
    Produces a cumulative gains chart and saves it locally.

    :param y_test: y_test series
    :param probability_predictions: dataframe of probability predictions, with the first column being the negative
    class predictions and the second column being the positive class predictions
    :param model_name: string name of the model
    """
    skplt.metrics.plot_cumulative_gain(y_test, probability_predictions)
    plt.savefig(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'cumulative_gains_plot.png'))
    plt.clf()


def plot_lift_curve_chart(y_test, probability_predictions, model_name):
    """
    Produces a lif curve and saves it locally.

    :param y_test: y_test series
    :param probability_predictions: dataframe of probability predictions, with the first column being the negative
    class predictions and the second column being the positive class predictions
    :param model_name: string name of the model
    """
    skplt.metrics.plot_lift_curve(y_test, probability_predictions)
    plt.savefig(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'lift_curve.png'))
    plt.clf()


def _assemble_negative_and_positive_pairs(y_test, probability_predictions, subset_percentage=0.1):
    """
    Finds the combination of every predicted probability in the negative class and every predicted probability in the
    positive class.

    :param y_test: y_test series
    :param probability_predictions: positive probability predictions series
    :param subset_percentage: percentage of observations to keep, as finding all the the combinations of positive and
    negative can result in a combinatorial explosion; default is 0.1
    :returns: list
    """
    df = pd.concat([y_test, probability_predictions], axis=1)
    df = df.sample(frac=subset_percentage)
    columns = list(df)
    true_label = columns[0]
    predicted_prob = columns[1]
    neg_df = df.loc[df[true_label] == 0]
    neg_probs = neg_df[predicted_prob].tolist()
    pos_df = df.loc[df[true_label] == 1]
    pos_probs = pos_df[predicted_prob].tolist()
    return list(itertools.product(neg_probs, pos_probs))


def _find_discordants(pairs):
    """
    Finds the number of discordants, defined as the number of cases where predicted probability in\\of the negative
    class observation is greater than the predicted probability of the positive class observation.

    :param pairs: tuple where the first element is the negative probability and the second element is the positive
    probability
    :returns: integer
    """
    discordants = 0
    if pairs[0] >= pairs[1]:
        discordants += 1
    return discordants


def find_concordant_discordant_ratio_and_somers_d(y_test, probability_predictions, model_name):
    """
    Finds the concordant-discordant ratiio and Somer's D and saved them locally

    :param y_test: y_test series
    :param probability_predictions: positive probability predictions series
    :param model_name: string name of the model
    """
    pairs = _assemble_negative_and_positive_pairs(y_test, probability_predictions)
    pool = mp.Pool(processes=mp.cpu_count())
    result = pool.map(_find_discordants, pairs)
    pairs = len(result)
    discordant_pairs = sum(result)
    concordant_discordant_ratio = 1 - (discordant_pairs / pairs)
    concordant_pairs = pairs - discordant_pairs
    somers_d = (concordant_pairs - discordant_pairs) / pairs
    pd.DataFrame({'concordant_discordant_ratio': [concordant_discordant_ratio], 'somers_d': [somers_d]}).to_csv(
        os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'concordant_discordant.csv'))


def run_mcnemar_test(y_test, model_1_class_predictions, model_2_class_predictions, model_1_name, model_2_name):
    """
    Runs the McNemar test to determine if there is a statistically significant difference in the class predictions.
    Writes the results and associated contingency table locally.

    :param y_test: y_test series
    :param model_1_class_predictions: class predictions from model 1
    :param model_2_class_predictions: class predictions from model 2
    :param model_1_name: name of the first model
    :param model_2_name: name of the second model
    """
    results_table = mcnemar_table(y_target=y_test, y_model1=model_1_class_predictions,
                                  y_model2=model_2_class_predictions)
    chi2, p = mcnemar(ary=results_table, corrected=True)
    pd.DataFrame({'chi2': [chi2], 'p': [p]}).to_csv(os.path.join(f'{model_1_name}_{model_2_name}_mcnemar_test.csv'))
    board = checkerboard_plot(results_table,
                              figsize=(6, 6),
                              fmt='%d',
                              col_labels=[f'{model_2_name} wrong', f'{model_2_name} right'],
                              row_labels=[f'{model_1_name} wrong', f'{model_1_name} right'])
    plt.tight_layout()
    plt.savefig(os.path.join(f'{model_1_name}_{model_2_name}_mcnemar_test.png'))
    plt.clf()


def run_cochran_q_test(y_test, *model_predictions, output_name):
    """
    Runs Cochran's Q test to determine if there is a statistically significant difference in more than two models' class
    predictions. The function can support up to five sets of predictions. Results are saved locally.

    :param y_test: y_test series
    :param model_predictions: arbitrary number of model predictions
    :param output_name: name to append to file to identify models used in the test
    """
    n_models = len(model_predictions)
    if n_models == 3:
        chi2, p = cochrans_q(y_test.values, model_predictions[0].values, model_predictions[1].values,
                             model_predictions[2].values)
    elif n_models == 4:
        chi2, p = cochrans_q(y_test.values, model_predictions[0].values, model_predictions[1].values,
                             model_predictions[2].values, model_predictions[3].values)
    elif n_models == 5:
        chi2, p = cochrans_q(y_test.values, model_predictions[0].values, model_predictions[1].values,
                             model_predictions[2].values, model_predictions[3].values, model_predictions[4].values)
    else:
        raise Exception('function cannot support more than five sets of predictions')
    pd.DataFrame({'chi2': [chi2], 'p': [p]}).to_csv(os.path.join(f'{output_name}_cochrans_q_test.csv'))


def produce_ks_statistic(y_test, probability_predictions, model_name):
    """
    Calculates the K-S statistic and saves the results locally.

    :param y_test: y_test series
    :param probability_predictions: dataframe of probability predictions, with the first column being the negative
    class predictions and the second column being the positive class predictions
    :param model_name: string name of the model
    """
    df = pd.concat([y_test, probability_predictions], axis=1)
    columns = list(df)
    true_label = columns[0]
    pos_predicted_prob = columns[1]
    pos_df = df.loc[df[true_label] == 1]
    neg_df = df.loc[df[true_label] == 0]
    result = ks_2samp(pos_df[pos_predicted_prob], neg_df[pos_predicted_prob])
    pd.DataFrame({'ks_statistic': [result[0]], 'p_value': [result[1]]}).to_csv(
        os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'ks_statistics.csv'), index=False)
    skplt.metrics.plot_ks_statistic(y_test, probability_predictions)
    plt.savefig(os.path.join(model_name, DIAGNOSTICS_DIRECTORY, 'ks_statistic.png'))
    plt.clf()


def perform_bias_variance_decomposition(model, x_train, y_train, x_test, y_test, model_name, n_boostraps=20):
    """
    Decomposes the average loss of a model into bias and variance. Writes out the results locally.

    :param model: trained model
    :param x_train: x_train
    :param y_train: y_train
    :param x_test: x_test
    :param y_test: y_test
    :param n_boostraps: number of bootstrap samples to take
    :param model_name: string name of the model
    """
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model, x_train, y_train, x_test, y_test,
                                                                loss='0-1_loss', random_seed=1234,
                                                                num_rounds=n_boostraps)
    pd.DataFrame({'avg_expected_loss': [avg_expected_loss], 'avg_bias': [avg_bias], 'avg_var': [avg_var]}).to_csv(
        os.path.join(model_name, DIAGNOSTICS_DIRECTORY, f'bias_variance_decomposition.csv'))


def run_omnibus_model_evaluation(pipeline, model_name, x_test, y_test, class_cutoff, target, evaluation_list,
                                 calibration_bins):
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
    """
    print(f'evaluating {model_name}...')
    predictions_df = produce_predictions(pipeline, model_name, x_test, y_test, class_cutoff)
    run_evaluation_metrics(predictions_df, target, model_name, evaluation_list)

    for metric in evaluation_list:
        get_bootstrap_estimate(y_test, predictions_df[metric[0]], metric[1], metric[2], model_name, samples=30)

    for n_bin in calibration_bins:
        plot_calibration_curve(y_test, predictions_df['1_prob'], n_bin, 'uniform', model_name)
        plot_calibration_curve(y_test, predictions_df['1_prob'], n_bin, 'quantile', model_name)

    optimal_threshold = find_optimal_class_cutoff(y_test, predictions_df['1_prob'], model_name)
    predictions_df['optimal_predicted_class'] = np.where(predictions_df['1_prob'] >= optimal_threshold, 1, 0)
    for predicted_class in ['predicted_class', 'optimal_predicted_class']:
        plot_confusion_matrix(y_test, predictions_df[predicted_class], model_name)

    produce_roc_curve_plot(pipeline, x_test, y_test, model_name)
    calculate_class_lift(y_test, predictions_df['predicted_class'], model_name)
    calculate_probability_lift(y_test, predictions_df['1_prob'], model_name)
    plot_cumulative_gains_chart(y_test, predictions_df[['0_prob', '1_prob']], model_name)
    plot_lift_curve_chart(y_test, predictions_df[['0_prob', '1_prob']], model_name)
    produce_ks_statistic(y_test, predictions_df[['0_prob', '1_prob']], model_name)
    find_concordant_discordant_ratio_and_somers_d(y_test, predictions_df['1_prob'], model_name)
