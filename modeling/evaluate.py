import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.calibration import calibration_curve

from modeling.config import DIAGNOSTICS_DIRECTORY


def produce_predictions(pipeline, model_name, x_test, y_test):
    """

    """
    df = pd.concat(
        [
        pd.DataFrame(pipeline.predict_proba(x_test), columns=['1_prob', '0_prob']),
        y_test.reset_index(drop=True),
        x_test.reset_index(drop=True)
        ],
        axis=1)
    df['predicted_class'] = np.where(df['1_prob'] >= 0.5, 1, 0)
    df = df[['predicted_class'] + [col for col in df.columns if col != 'predicted_class']]
    df.to_csv(os.path.join(DIAGNOSTICS_DIRECTORY, f'{model_name}_predictions.csv'), index=False)
    return df


def _evaluate_model(df, target, predictions, scorer, metric_name):
    """

    """
    score = scorer(df[target], df[predictions])
    df = pd.DataFrame({metric_name: [score]})
    return df


def run_evaluation_metrics(df, target, model_name, evaluation_list):
    """

    """
    main_df = pd.DataFrame()
    for metric_config in evaluation_list:
        temp_df = _evaluate_model(df, target, metric_config[0], metric_config[1], metric_config[2])
        main_df = pd.concat([main_df, temp_df], axis=1)
    main_df.to_csv(os.path.join(DIAGNOSTICS_DIRECTORY, f'{model_name}_evaluation.csv'), index=False)


def plot_calibration_curve(y_test, predictions, n_bins, model_name):
    """

    """
    y, x = calibration_curve(y_test, predictions, n_bins=n_bins)
    fig, ax = plt.subplots()
    plt.plot(x, y, marker='o', linewidth=1, label='model')
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle('Calibration Plot')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability in Each Bin')
    plt.legend()
    plt.savefig(os.path.join(DIAGNOSTICS_DIRECTORY, f'{model_name}_calibration_plot.png'))
    plt.clf()


def produce_shap_values(pipeline, x_test, model_name):
    """

    """
    model = pipeline.named_steps['model']
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap_df = pd.DataFrame(shap_values, columns=list(x_test))
    shap_df.to_csv(os.path.join(DIAGNOSTICS_DIRECTORY, f'{model_name}_shap_values.csv'), index=False)
    shap_expected_value = pd.DataFrame(explainer.expected_value[0], columns=['expected_value'])
    shap_expected_value.to_csv(os.path.join(DIAGNOSTICS_DIRECTORY, f'{model_name}_shap_expected.csv'), index=False)
    global_shap = np.abs(shap_values).mean(0)
    global_shap_df = pd.DataFrame(list(zip(x_test.columns, global_shap)), columns=['column', 'shap_value'])
    global_shap_df.sort_values(by=['shap_value'], ascending=False, inplace=True)
    global_shap_df.to_csv(os.path.join(DIAGNOSTICS_DIRECTORY, f'{model_name}_shap_global.csv'), index=False)
    shap.summary_plot(shap_values, x_test, show=False)
    plt.savefig(os.path.join(DIAGNOSTICS_DIRECTORY, f'{model_name}_shap_values.png'), bbox_inches='tight')
    plt.clf()
    shap.summary_plot(shap_values, x_test, plot_type='bar', show=False)
    plt.savefig(os.path.join(DIAGNOSTICS_DIRECTORY, f'{model_name}_shap_values_summary.png'), bbox_inches='tight')
    plt.clf()
