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

from helpers.model_helpers import make_directories_if_not_exists
from data.db import log_model_scores_to_mysql


def produce_predictions(estimator, model_uid, x_test, y_test, class_cutoff):
    """
    Produces a dataframe consisting of the probability and class predictions from the model on the test set along
    with the y_test. The dataframe is saved locally.

    :param estimator: estimator object
    :param model_uid: model uid
    :param x_test: x_test dataframe
    :param y_test: y_test series
    :param class_cutoff: the probability cutoff to separate classes
    :returns: pandas dataframe
    """
    df = pd.concat(
        [
            pd.DataFrame(estimator.predict_proba(x_test), columns=['0_prob', '1_prob']),
            y_test.reset_index(drop=True)
        ],
        axis=1)
    df['predicted_class'] = np.where(df['1_prob'] >= class_cutoff, 1, 0)
    df = df[['predicted_class'] + [col for col in df.columns if col != 'predicted_class']]
    df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'predictions', 'predictions_vs_actuals.csv'),
              index=False)
    return df


def _evaluate_model(target_series, prediction_series, scorer, metric_name):
    """
    Applies a scorer function to evaluate predictions against the ground-truth labels.

    :param target_series: target series
    :param prediction_series: prediction series
    :param scorer: scoring function to evaluate the predictions
    :param metric_name: name of the metric we are using to score our model

    :returns: pandas dataframe
    """
    score = scorer(target_series, prediction_series)
    df = pd.DataFrame({metric_name: [score]})
    return df


def run_and_save_evaluation_metrics(df, target, model_uid, evaluation_list, db_schema_name, db_conn, log_to_db):
    """
    Runs a series of evaluations metrics on a model's predictions and writes the results locally.

    :param df: pandas dataframe containing the predictions and the actuals
    :param target: name of the target column
    :param model_uid: model uid
    :param evaluation_list: list of named tuples, which each tuple having the ordering of: the column with the
    predictions, the scoring function callable, and the name of the metric
    :param db_schema_name: database schema to log metrics to
    :param db_conn: database connection
    :param log_to_db: Boolean of whether to log metrics to the database
    """
    main_df = pd.DataFrame()
    for evaluation_config in evaluation_list:
        temp_df = _evaluate_model(df[target], df[evaluation_config.evaluation_column],
                                  evaluation_config.scorer_callable, evaluation_config.metric_name)
        main_df = pd.concat([main_df, temp_df], axis=1)
    main_df = main_df.T
    main_df.reset_index(inplace=True)
    main_df.columns = ['scoring_metric', 'holdout_score']
    main_df['model_uid'] = model_uid
    main_df['holdout_type'] = 'test'
    if log_to_db:
        log_model_scores_to_mysql(main_df, db_schema_name, db_conn)
    main_df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_files', 'evaluation_scores.csv'),
                   index=False)


def evaluate_model_by_partition(x_df, target, predictions_df, evaluation_list, model_uid, numeric_bins, drop_cols):
    """
    Runs evaluation metrics on different subsets of the data. If the data is numeric, it is binned by the number of
    numeric_bins. Results are written as a csv

    :param x_df: x dataframe
    :param target: name of target
    :param predictions_df: dataframe of predictions vs. actuals
    :param evaluation_list: list of named tuples, which each tuple having the ordering of: the column with the
    predictions, the scoring function callable, and the name of the metric
    :param model_uid: model uid
    :param numeric_bins: number of bins for numeric features
    :param drop_cols: list of columns to drop
    """
    x_df = x_df.reset_index(drop=True)
    categorical_cols = list(x_df.select_dtypes(include='object'))
    numeric_cols = list(x_df.select_dtypes(include='number'))
    predictions_df = pd.concat([predictions_df, x_df], axis=1)

    main_df = pd.DataFrame()
    for col in categorical_cols + numeric_cols:
        if col not in drop_cols:
            if col in numeric_cols:
                predictions_df[col] = pd.qcut(predictions_df[col], numeric_bins)
                predictions_df[col] = predictions_df[col].astype(str)
            levels = list(predictions_df[col].unique())
            for level in levels:
                temp_predictions_df = predictions_df.loc[predictions_df[col] == level]
                classes = list(temp_predictions_df[target].unique())
                if len(classes) >= 2:
                    main_level_df = pd.DataFrame()
                    for evaluation_config in evaluation_list:
                        temp_df = _evaluate_model(temp_predictions_df[target],
                                                  temp_predictions_df[evaluation_config.evaluation_column],
                                                  evaluation_config.scorer_callable, evaluation_config.metric_name)
                        main_level_df = pd.concat([main_level_df, temp_df], axis=1)

                    main_level_df = main_level_df.T
                    main_level_df['partition'] = f'{col}_{level}'
                    main_level_df['observations'] = len(temp_predictions_df)
                    main_level_df.reset_index(inplace=True)
                    main_level_df.rename(columns={0: 'score', 'index': 'metric'}, inplace=True)
                    main_df = main_df.append(main_level_df)

    main_df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_files',
                                'evaluation_scores_by_partition.csv'), index=False)


def plot_calibration_curve(y_test, predictions, n_bins, bin_strategy, model_uid):
    """
    Produces a calibration plot and saves it locally. The raw data behind the plot is also written locally.

    :param y_test: y_test series
    :param predictions: predictions series
    :param n_bins: number of bins for the predictions
    :param bin_strategy: uniform - all bins have the same width; quantile - bins have the same number of observations
    :param model_uid: model uid
    """
    try:
        prob_true, prob_pred = calibration_curve(y_test, predictions, n_bins=n_bins, strategy=bin_strategy)
        fig, ax = plt.subplots()
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='model')
        line = mlines.Line2D([0, 1], [0, 1], color='black')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        fig.suptitle(f' {bin_strategy.title()} Calibration Plot {n_bins} Requested Bins')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability in Each Bin')
        plt.legend()
        plt.savefig(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_plots',
                                 f'{bin_strategy}_{n_bins}_calibration_plot.png'))
        plt.clf()
        calibration_df = pd.DataFrame({'prob_true': prob_true, 'prob_pred': prob_pred})
        calibration_df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_files',
                                           f'{bin_strategy}_{n_bins}_calibration_summary.csv'), index=False)
    except Exception as e:
        print(e)


def plot_distribution_of_positive_predictions(model_uid, positive_predictions_series):
    """
    Makes a density plot of the positive predictions.

    :param model_uid: model uid
    :param positive_predictions_series: series of positive predictions
    """
    sns.kdeplot(positive_predictions_series, shade=True, color='b', legend=False)
    plt.xlabel('positive prediction space')
    plt.ylabel('density')
    plt.title('Positive Predictions Distribution')
    plt.savefig(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_plots', 'pos_prediction_distribution.png'))
    plt.clf()


def produce_roc_curve_plot(estimator, x_test, y_test, model_uid):
    """
    Produces a ROC curve plot and saves the result locally.

    :param estimator: estimator object
    :param x_test: x_test dataframe
    :param y_test: y_test series
    :param model_uid: model uid
    """
    plot_roc_curve(estimator, x_test, y_test)
    plt.savefig(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_plots', 'roc_curve.png'))
    plt.clf()


def find_optimal_class_cutoff(y_test, probability_predictions, model_uid):
    """
    Finds the optimal class cutoff based on the ROC curve and saves the result locally.

    :param y_test: y_test series
    :param probability_predictions: probability predictions for the positive class
    :param model_uid: model uid
    :return: float that represents the optimal threshold
    """
    fpr, tpr, threshold = roc_curve(y_test, probability_predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    best_tpr = tpr[optimal_idx]
    best_fpr = fpr[optimal_idx]
    df = pd.DataFrame({'optimal_threshold': [optimal_threshold], 'best_tpr': [best_tpr], 'best_fpr': [best_fpr]})
    df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_files', 'optimal_class_cutoff.csv'),
              index=False)
    return optimal_threshold


def plot_confusion_matrix(y_test, class_predictions, model_uid):
    """
    Plots a confusion matrix ans saves it locally.

    :param y_test: y_test series
    :param class_predictions: class predictions series
    :param model_uid: model uid
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
    plt.savefig(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_plots', 'confusion_matrix.png'))
    plt.clf()


def get_bootstrap_estimate(y_test, prediction_series, scoring_metric, metric_name, model_uid, samples=30):
    """
    Produces a bootstrap estimate for a scoring metric. y_test and it's associated predictions are sampled with
    replacement. The samples argument dictates how many times to perform each sampling. Each sample is of equal length,
    determined by len(y_test) / samples. Summary statistics and a density plot of the distribution are saved locally.

    :param y_test: y_test series
    :param prediction_series: relevant prediction series
    :param scoring_metric: scikit-learn scoring metric callable (e.g. roc_auc_score)
    :param metric_name: string name of the metric
    :param model_uid: model uid
    :param samples: number of samples to take; default is 30
    """
    y_test = y_test.reset_index(drop=True)
    prediction_series = prediction_series.reset_index(drop=True)
    df = pd.concat([y_test, prediction_series], axis=1)
    approx_sample_size = int(len(df) / samples)
    metrics_df = pd.DataFrame()

    for sample in range(samples):
        temp_df = df.sample(n=approx_sample_size)
        try:
            score = scoring_metric(temp_df.iloc[:, 0], temp_df.iloc[:, 1], labels=[0, 1])
        except:
            score = scoring_metric(temp_df.iloc[:, 0], temp_df.iloc[:, 1])
        temp_df = pd.DataFrame({'score': [score]})
        metrics_df = metrics_df.append(temp_df)
    described_df = metrics_df.describe()
    described_df.to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_files',
                                     f'{metric_name}_sample_scores.csv'))

    sns.kdeplot(metrics_df['score'], shade=True, color='b', legend=False)
    plt.xlabel(metric_name)
    plt.ylabel('density')
    plt.title(f'{metric_name} distribution')
    plt.savefig(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_plots',
                             f'{metric_name}_density_plot.png'))
    plt.clf()


def calculate_class_lift(y_test, class_predictions, model_uid):
    """
    Calculates the lift of a model, based on predicted class labels.

    :param y_test: y_test series
    :param class_predictions: class predictions series
    :param model_uid: model uid
    """
    lift = lift_score(y_test, class_predictions)
    pd.DataFrame({'lift': [lift]}).to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_plots',
                                                       'class_lift.csv'), index=False)


def calculate_probability_lift(y_test, probability_predictions, model_uid, n_bins=10):
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
    :param model_uid: model uid
    :param n_bins: number of bins to segment the probability predictions
    """
    y_test = y_test.reset_index(drop=True)
    prediction_series = probability_predictions.reset_index(drop=True)
    df = pd.concat([y_test, prediction_series], axis=1)
    columns = list(df)
    class_col = columns[0]
    proba_col = columns[1]
    base_rate = df[class_col].mean()

    df['1_prob_bin'] = pd.qcut(df[proba_col], q=n_bins, labels=list(range(1, n_bins + 1)))
    grouped_df = df.groupby('1_prob_bin').agg({proba_col: 'mean', class_col: 'mean'})
    grouped_df.reset_index(inplace=True)
    grouped_df['1_prob_diff'] = grouped_df[proba_col] - grouped_df[class_col]
    grouped_df['base_rate_diff'] = base_rate - grouped_df[class_col]

    prob_diff = grouped_df['1_prob_diff'].abs().sum()
    base_rate_diff = grouped_df['base_rate_diff'].abs().sum()
    lift = base_rate_diff / prob_diff
    pd.DataFrame({'lift': [lift]}).to_csv(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_files',
                                                       'proba_lift.csv'), index=False)
    return lift


def plot_cumulative_gains_chart(y_test, probability_predictions, model_uid):
    """
    Produces a cumulative gains chart and saves it locally.

    :param y_test: y_test series
    :param probability_predictions: dataframe of probability predictions, with the first column being the negative
    class predictions and the second column being the positive class predictions
    :param model_uid: model uid
    """
    skplt.metrics.plot_cumulative_gain(y_test, probability_predictions)
    plt.savefig(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_plots', 'cumulative_gains_plot.png'))
    plt.clf()


def plot_lift_curve_chart(y_test, probability_predictions, model_uid):
    """
    Produces a lif curve and saves it locally.

    :param y_test: y_test series
    :param probability_predictions: dataframe of probability predictions, with the first column being the negative
    class predictions and the second column being the positive class predictions
    :param model_uid: model uid
    """
    skplt.metrics.plot_lift_curve(y_test, probability_predictions)
    plt.savefig(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_plots', 'lift_curve.png'))
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


def find_concordant_discordant_ratio_and_somers_d(y_test, probability_predictions, model_uid):
    """
    Finds the concordant-discordant ratiio and Somer's D and saved them locally

    :param y_test: y_test series
    :param probability_predictions: positive probability predictions series
    :param model_uid: model uid
    """
    pairs = _assemble_negative_and_positive_pairs(y_test, probability_predictions)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.map(_find_discordants, pairs)
    pairs = len(result)
    discordant_pairs = sum(result)
    concordant_discordant_ratio = 1 - (discordant_pairs / pairs)
    concordant_pairs = pairs - discordant_pairs
    somers_d = (concordant_pairs - discordant_pairs) / pairs
    pd.DataFrame({'concordant_discordant_ratio': [concordant_discordant_ratio], 'somers_d': [somers_d]}).to_csv(
        os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_files', 'concordant_discordant.csv'),
        index=False)


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
    plt.savefig(os.path.join('modeling', 'comparison_files', f'{model_1_name}_{model_2_name}_mcnemar_test.png'))
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
    pd.DataFrame({'chi2': [chi2], 'p': [p]}).to_csv(os.path.join('modeling', 'comparison_files',
                                                                 f'{output_name}_cochrans_q_test.csv'))


def produce_ks_statistic(y_test, probability_predictions, model_uid):
    """
    Calculates the K-S statistic and saves the results locally.

    :param y_test: y_test series
    :param probability_predictions: dataframe of probability predictions, with the first column being the negative
    class predictions and the second column being the positive class predictions
    :param model_uid: model uid
    """
    df = pd.concat([y_test, probability_predictions], axis=1)
    columns = list(df)
    true_label = columns[0]
    pos_predicted_prob = columns[1]
    pos_df = df.loc[df[true_label] == 1]
    neg_df = df.loc[df[true_label] == 0]
    result = ks_2samp(pos_df[pos_predicted_prob], neg_df[pos_predicted_prob])
    pd.DataFrame({'ks_statistic': [result[0]], 'p_value': [result[1]]}).to_csv(
        os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_files', 'ks_statistics.csv'), index=False)
    skplt.metrics.plot_ks_statistic(y_test, probability_predictions)
    plt.savefig(os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_plots', 'ks_statistic.png'))
    plt.clf()


def perform_bias_variance_decomposition(estimator, x_train, y_train, x_test, y_test, model_uid, n_boostraps=20):
    """
    Decomposes the average loss of a model into bias and variance. Writes out the results locally.

    :param estimator: estimator object
    :param x_train: x_train
    :param y_train: y_train
    :param x_test: x_test
    :param y_test: y_test
    :param n_boostraps: number of bootstrap samples to take
    :param model_uid: model uid
    """
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(estimator, x_train, y_train, x_test, y_test,
                                                                loss='0-1_loss', random_seed=1234,
                                                                num_rounds=n_boostraps)
    pd.DataFrame({'avg_expected_loss': [avg_expected_loss], 'avg_bias': [avg_bias], 'avg_var': [avg_var]}).to_csv(
        os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_files', f'bias_variance_decomposition.csv'),
        index=False)


def run_omnibus_model_evaluation(estimator, model_uid, x_test, y_test, class_cutoff, target, evaluation_list,
                                 calibration_bins, db_schema_name=None, db_conn=None, log_to_db=False):
    """
    Runs a series of functions to evaluate a model's performance.

    :param estimator: estimator object
    :param model_uid: model uid
    :param x_test: x_test dataframe
    :param y_test: y_test series
    :param class_cutoff: the probability cutoff to separate classes
    :param target: name of the target column
    :param evaluation_list: list of tuples, which each tuple having the ordering of: the column with the predictions,
    the scoring function callable, and the name of the metric
    :param calibration_bins: number of bins in the calibration plot
    :param db_schema_name: database schema to log metrics to
    :param db_conn: database connection
    :param log_to_db: Boolean of whether to log metrics to the database
    """
    print(f'evaluating {model_uid}...')
    make_directories_if_not_exists([
        os.path.join('modeling', model_uid, 'diagnostics', 'predictions'),
        os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_files'),
        os.path.join('modeling', model_uid, 'diagnostics', 'evaluation_plots'),
        os.path.join('modeling', 'comparison_files')
    ])

    predictions_df = produce_predictions(estimator, model_uid, x_test, y_test, class_cutoff)
    run_and_save_evaluation_metrics(predictions_df, target, model_uid, evaluation_list, db_schema_name, db_conn,
                                    log_to_db)
    plot_distribution_of_positive_predictions(model_uid, predictions_df['1_prob'])

    for metric in evaluation_list:
        get_bootstrap_estimate(y_test, predictions_df[metric.evaluation_column], metric.scorer_callable,
                               metric.metric_name, model_uid, samples=30)

    for n_bin in calibration_bins:
        plot_calibration_curve(y_test, predictions_df['1_prob'], n_bin, 'uniform', model_uid)
        plot_calibration_curve(y_test, predictions_df['1_prob'], n_bin, 'quantile', model_uid)

    optimal_threshold = find_optimal_class_cutoff(y_test, predictions_df['1_prob'], model_uid)
    predictions_df['optimal_predicted_class'] = np.where(predictions_df['1_prob'] >= optimal_threshold, 1, 0)
    for predicted_class in ['predicted_class', 'optimal_predicted_class']:
        plot_confusion_matrix(y_test, predictions_df[predicted_class], model_uid)

    produce_roc_curve_plot(estimator, x_test, y_test, model_uid)
    calculate_class_lift(y_test, predictions_df['predicted_class'], model_uid)
    calculate_probability_lift(y_test, predictions_df['1_prob'], model_uid)
    plot_cumulative_gains_chart(y_test, predictions_df[['0_prob', '1_prob']], model_uid)
    plot_lift_curve_chart(y_test, predictions_df[['0_prob', '1_prob']], model_uid)
    produce_ks_statistic(y_test, predictions_df[['0_prob', '1_prob']], model_uid)
    find_concordant_discordant_ratio_and_somers_d(y_test, predictions_df['1_prob'], model_uid)
