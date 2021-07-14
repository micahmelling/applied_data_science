import os
import shap
import joblib
import matplotlib.pyplot as plt
import random

from helpers.model_helpers import make_directories_if_not_exists
from modeling.explain import transform_data_with_pipeline


def make_waterfall_plots(indices, expected_value, shap_values, x_df):
    """
    Makes a SHAP waterfall plot for each index passed through for x_df. Plots are saved to the shap_plots directory.

    :param indices: the indices of x_df for which to make plots
    :param expected_value: the expected value produced from the explainer
    :param shap_values: numpy array of shap values
    :param x_df: x dataframe
    """
    for index in indices:
        shap.waterfall_plot(expected_value, shap_values[index], x_df.iloc[index], show=False)
        plt.savefig(os.path.join('shap_plots', f'shap_waterfall_{index}.png'), bbox_inches='tight')
        plt.clf()


def make_decision_plot(expected_value, shap_values, x_df):
    """
    Makes a SHAP decision plot and saved it ont the shap_plots directory.

    :param expected_value: the expected value produced from the explainer
    :param shap_values: numpy array of shap values
    :param x_df: x dataframe
    """
    shap.decision_plot(expected_value, shap_values, x_df, show=False, link='identity')
    plt.savefig(os.path.join('shap_plots', 'shap_decision_plot.png'), bbox_inches='tight')
    plt.clf()


def make_shap_interaction_plots(shap_values, x_df, n_rank):
    """
    Plots a series pf SHAP interaction plots fo the top n_rank features. The interaction plot shows the most meaningful
    interaction with feature.

    :param shap_values: numpy array of shap values
    :param x_df: x dataframe
    :param n_rank: number of top ranking features for which to plot interactions
    """
    for rank in range(n_rank):
        shap.dependence_plot(f'rank({rank})', shap_values, x_df, show=False)
        plt.savefig(os.path.join('shap_plots', f'shap_interaction_{rank}.png'), bbox_inches='tight')
        plt.clf()


def main():
    make_directories_if_not_exists(['shap_plots'])
    x_df = joblib.load('../modeling/extra_trees_uncalibrated_202101312152421899080600/data/x_test.pkl')
    x_df = x_df.sample(n=500)
    pipeline = joblib.load('../modeling/extra_trees_uncalibrated_202101312152421899080600/models/model.pkl')
    model = pipeline.named_steps['model']
    x_df = transform_data_with_pipeline(pipeline, x_df)
    x_df.reset_index(inplace=True, drop=True)
    x_indices = x_df.index.to_list()
    random_x_indices = random.sample(x_indices, 3)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_df)[1]
    expected_value = explainer.expected_value[1]
    make_waterfall_plots(random_x_indices, expected_value, shap_values, x_df)
    make_decision_plot(expected_value, shap_values, x_df)
    make_shap_interaction_plots(shap_values, x_df, 10)


if __name__ == "__main__":
    main()
