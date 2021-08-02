import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
import ppscore as pps
import operator

from copy import deepcopy
from tqdm import tqdm
from functools import partial
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import fpgrowth, association_rules

from data.db import make_mysql_connection


warnings.filterwarnings('ignore')


def create_exploration_directories(images_path, files_path):
    """
    Creates images_path and files_path to store exploration output:

    :param images_path: path in which to store images
    :param files_path: path in which to store files
    """
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(files_path):
        os.makedirs(files_path)


def create_tsne_visualization(df, target, save_path, sample_size=10_000):
    """
    Creates a t-SNE visualization and saves it into IMAGES_PATH. The visualization will help us visualize our entire
    dataset and will highlight the data points in each class. This will allow us to see how clustered or interspersed
    our target classes are.

    :param df: pandas dataframe
    :param target: name of the target
    :param sample_size: number of observations to sample since t-SNE is computationally expensive; default is 10_000
    :param save_path: path in which to save the output
    """
    print('creating tsne visualization...')
    df = df.sample(n=sample_size)
    target_df = df[[target]]
    df = df.drop(target, 1)
    df = df.select_dtypes(include=['float64', 'float32', 'int'])
    df.dropna(how='all', inplace=True, axis=1)
    df = pd.DataFrame(SimpleImputer(strategy='mean', copy=False).fit_transform(df), columns=list(df))
    df = pd.DataFrame(StandardScaler().fit_transform(df), columns=list(df))

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df)
    target_df['tsne_2d_one'] = tsne_results[:, 0]
    target_df['tsne_2d_two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne_2d_one",
        y="tsne_2d_two",
        palette=sns.color_palette("hls", 2),
        data=target_df,
        hue=target,
        legend="full",
        alpha=0.3
    )
    plt.title('TSNE Plot')
    plt.savefig(os.path.join(save_path, 'tsne.png'))
    plt.clf()


def analyze_categorical_feature_dispersion(df, feature, fill_na_value='unknown'):
    """
    Finds the percentage of observations for each categorical level within a column.

    :param df: pandas dataframe
    :param feature: name of the feature to analyze
    :param fill_na_value: value to fill nulls; default is 'unknown'
    :returns: pandas dataframe
    """
    df.fillna({feature: fill_na_value}, inplace=True)
    total_observations = len(df)
    grouped_df = pd.DataFrame(df.groupby(feature)[feature].count())
    grouped_df.columns = ['total_count']
    grouped_df.reset_index(inplace=True)
    grouped_df['percentage_of_category_level'] = grouped_df['total_count'] / total_observations
    grouped_df = grouped_df[[feature, 'total_count', 'percentage_of_category_level']]
    grouped_df.rename(columns={feature: 'feature'}, inplace=True)
    grouped_df['feature'] = feature + '_' + grouped_df['feature']
    grouped_df['category'] = feature
    return grouped_df


def make_density_plot_by_binary_target(df, feature, target, save_path):
    """
    Creates an overlayed density plot. One density plot for the feature is plotted for the first target level. Then a
    second density plot for the feature is plotted for the second target level. The plot is saved to IMAGES_PATH.
    Per the name of the function, this only expects a binary target.

    :param df: pandas dataframe
    :param feature: name of the feature to plot
    :param target: name of the target
    :param save_path: path in which to save the output
    """
    target_levels = list(df[target].unique())
    target_df = df[[target]]
    target_df.reset_index(inplace=True, drop=True)
    df = df[[feature]]
    df.reset_index(inplace=True, drop=True)
    df = pd.DataFrame(SimpleImputer(strategy='mean', copy=False).fit_transform(df), columns=list(df))
    df = pd.concat([target_df, df], axis=1)

    df_level0 = df.loc[df[target] == target_levels[0]]
    df_level0.rename(columns={feature: target_levels[0]}, inplace=True)
    p0 = sns.kdeplot(df_level0[target_levels[0]], shade=True, color='r', legend=True)
    df_level1 = df.loc[df[target] == target_levels[1]]
    df_level1.rename(columns={feature: target_levels[1]}, inplace=True)
    p1 = sns.kdeplot(df_level1[target_levels[1]], shade=True, color='b', legend=True)
    plt.xlabel(feature)
    plt.ylabel('density')
    plt.savefig(os.path.join(save_path, f'density_plot_by_target_for_{feature}.png'))
    plt.clf()


def calculate_binary_target_balance(df, target):
    """
    Calculates the target class balances.

    :param df: pandas dataframe
    :param target: name of the target
    :returns: tuple containing the target level names and their respective percentage of total observations. the tuple
    contains two items, each a list. in each list, the first item is the class name and the second item is the
    percentage of observations
    """
    total_observations = len(df)
    grouped_df = pd.DataFrame(df.groupby(target)[target].count())
    grouped_df.columns = ['count']
    grouped_df.reset_index(inplace=True)
    grouped_df['percentage'] = grouped_df['count'] / total_observations
    grouped_df = grouped_df[[target, 'percentage']]
    target_balance_tuple = [tuple(x) for x in grouped_df.values]
    return target_balance_tuple


def analyze_category_by_binary_target(df, feature, target, target_balance_tuple, fill_na_value='unknown'):
    """
    For the categorical feature provided, for each of its levels, find the difference between the actual and expected
    percentages of the positive class.

    :param df: pandas dataframe
    :param feature: name of the feature to analyze
    :param target: name of the target
    :param target_balance_tuple: tuple that contains two items, each a list. in each list, the first item is the class
    name and the second item is the percentage of observations
    :param fill_na_value: value to fill nulls; default is 'unknown'
    """
    class_0_percent = target_balance_tuple[0][1]
    class_1_name = target_balance_tuple[1][0]
    class_1_percent = target_balance_tuple[1][1]

    df.fillna({feature: fill_na_value}, inplace=True)
    grouped_df = pd.DataFrame(df.groupby([feature, target])[feature].count())
    grouped_df.columns = ['count']
    grouped_df.reset_index(inplace=True)

    feature_count_df = pd.DataFrame(df.groupby(feature)[feature].count())
    feature_count_df.columns = ['feature_count']
    feature_count_df.reset_index(inplace=True)
    grouped_df = pd.merge(grouped_df, feature_count_df, how='inner', on=feature)
    grouped_df['actual_percentage'] = grouped_df['count'] / grouped_df['feature_count']

    grouped_df['class_1_flag'] = np.where(grouped_df[target] == class_1_name, 1, 0)
    grouped_df['expected_observations'] = np.where(grouped_df['class_1_flag'] == 1,
                                                   grouped_df['feature_count'] * class_1_percent,
                                                   grouped_df['feature_count'] * class_0_percent)
    grouped_df['expected_percentage'] = grouped_df['expected_observations'] / grouped_df['feature_count']
    grouped_df['diff_from_expectation'] = grouped_df['actual_percentage'] - grouped_df['expected_percentage']
    grouped_df = grouped_df.loc[grouped_df[target] == class_1_name]
    grouped_df.rename(columns={feature: 'feature', 'count': 'positive_count'}, inplace=True)
    grouped_df['feature'] = feature + '_' + grouped_df['feature']
    grouped_df = grouped_df[['feature', 'positive_count', 'diff_from_expectation']]
    return grouped_df


def plot_category_level_counts_and_target_connection(connection_df, dispersion_df, save_path):
    """
    Plots the difference between the actual and expected percentages of the positive class (diff_from_expectation)
    along with the percentage of observations for each category level (percentage_of_category_level).

    :param connection_df: dataframe produced by analyze_categorical_feature_dispersion
    :param dispersion_df: dataframe produced by analyze_category_by_binary_target
    :param save_path: path in which to save the output
    """
    merged_df = pd.merge(dispersion_df, connection_df, how='left', on='feature')
    merged_df.fillna(value=0, inplace=True)
    merged_df = pd.melt(merged_df, id_vars=['feature', 'category'])
    merged_df = merged_df.loc[merged_df['variable'].isin(['percentage_of_category_level', 'diff_from_expectation'])]

    feature_categories = list(merged_df['category'].unique())
    for category in feature_categories:
        temp_df = merged_df.loc[merged_df['category'] == category]
        sns.factorplot(x='feature', y='value', hue='variable', data=temp_df, kind='bar', legend=False)
        plt.title('Category Connection Summary for ' + category)
        plt.xticks(rotation=90)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'category_connection_summary_for_{category}.png'))
        plt.clf()


def generate_summary_statistics(df, target, save_path):
    """
    Calculates summary statistics for each target level and writes the results into FILES_PATH.

    :param df: pandas dataframe
    :param target: name of the target
    :param save_path: path in which to save the output
    """
    total_df = df.describe(include='all')
    total_df['level'] = 'total'
    for level in (list(set(df[target].tolist()))):
        temp_df = df.loc[df[target] == level]
        temp_df = temp_df.describe(include='all')
        temp_df['level'] = level
        total_df = total_df.append(temp_df)
    total_df.to_csv(os.path.join(save_path, 'summary_statistics.csv'), index=True)


def find_highly_correlated_features(df, save_path, correlation_cutoff=0.98):
    """
    Finds the correlation among all features in a dataset and flags those that are highly correlated. Output is saved
    into FILES_PATH. This will produce some false positives for dummy-coded features.

    :param df: pandas dataframe
    :param correlation_cutoff: cutoff for how high a correlation must be to be flagged; default is 0.98
    :param save_path: path in which to save the output
    """
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.DataFrame(df.corr().abs().unstack())
    df.reset_index(inplace=True)
    df.columns = ['feature1', 'feature2', 'correlation']
    df.sort_values(by=['correlation', 'feature1', 'feature2'], ascending=False, inplace=True)
    df['lag_feature1'] = df['feature1'].shift(-1)
    df['lag_feature2'] = df['feature2'].shift(-1)
    df['duplication'] = np.where((df['feature2'] == df['lag_feature1']) &
                                 (df['feature1'] == df['lag_feature2']),
                                 'yes', 'no')
    df = df.loc[df['feature1'] != df['feature2']]
    df = df.loc[df['duplication'] == 'no']
    df.drop(['duplication', 'lag_feature1', 'lag_feature2'], 1, inplace=True)
    df['high_correlation'] = np.where(df['correlation'] >= correlation_cutoff, 'yes', 'no')
    df.to_csv(os.path.join(save_path, 'feature_correlation.csv'), index=False)


def score_features_using_mutual_information(df, target, save_path):
    """
    Scores univariate features using mutual information. Saves the output locally.

    :param df: pandas dataframe
    :param target: name of the target
    :param save_path: path in which to save the output
    """
    print('scoring features using mutual information...')
    y = df[target]
    x = df.drop([target], axis=1)
    x_numeric = x.select_dtypes(include='number')
    x_numeric.dropna(how='all', inplace=True, axis=1)
    x_numeric = pd.DataFrame(SimpleImputer(strategy='mean', copy=False).fit_transform(x_numeric),
                             columns=list(x_numeric))
    x_categorical = x.select_dtypes(include='object')
    x_categorical = pd.get_dummies(x_categorical, dummy_na=True)

    def _fit_feature_selector(x, scorer, discrete_features):
        scorer = partial(scorer, discrete_features=discrete_features)
        feature_selector = SelectPercentile(scorer)
        _ = feature_selector.fit_transform(x, y)
        feature_scores = pd.DataFrame()
        feature_scores['score'] = feature_selector.scores_
        feature_scores['attribute'] = x.columns
        return feature_scores

    numeric_scores = _fit_feature_selector(x_numeric, mutual_info_classif, discrete_features=False)
    categorical_scores = _fit_feature_selector(x_categorical, mutual_info_classif, discrete_features=True)
    feature_scores = pd.concat([numeric_scores, categorical_scores])
    feature_scores.reset_index(inplace=True, drop=True)
    feature_scores.sort_values(by='score', ascending=False, inplace=True)
    feature_scores.to_csv(os.path.join(save_path, 'univariate_features_mutual_information.csv'), index=False)


def run_predictive_power_score(df, target, save_path):
    """
    Calculates the predictive power score (pps) for each feature. If the score is 0, then it is not any better than a
    baseline model. If it's 1, then the feature is a perfect predictor. The model_score is the weighted F1 score for
    a univariate model predicting the target.

    :param df: pandas dataframe
    :param target: name of our target
    :param save_path: path in which to save the output
    """
    df = pd.get_dummies(df, dummy_na=True)
    df[target] = df[target].astype(str)
    pps_df = pd.DataFrame({})
    for feature in tqdm(list(df)):
        if feature != target:
            temp_score_dict = pps.score(df, feature, target)
            temp_ppscore = temp_score_dict.get('ppscore')
            temp_model_score = temp_score_dict.get('model_score')
            temp_df = pd.DataFrame({'feature': [feature], 'pps': [temp_ppscore], 'model_score': [temp_model_score]})
            pps_df = pps_df.append(temp_df)
    pps_df.to_csv(os.path.join(save_path, 'predictive_power_score.csv'), index=False)


def run_kmeans_clustering(df, drop_list, max_clusters, fill_na_value=0, samples=25_000):
    """
    Runs a k-means clustering algorithm to assign each observation to a cluster.

    :param df: pandas dataframe we want to run the clustering algorithm on
    :param drop_list: features we want to exclude from clustering
    :param max_clusters: the maximum number of clusters to potentially have
    :param fill_na_value: value to fill for missing numeric values
    :param samples: Since k-means can be computationally expensive, we might want to only run it on a subset of data
    :returns: pandas dataframe that can be used in get_cluster_summary()
    """
    print('running k-means clustering...')
    append_df = deepcopy(df)
    append_df = append_df.sample(n=samples)
    append_df = pd.get_dummies(append_df, dummy_na=True)
    append_df.fillna(value=fill_na_value, inplace=True)
    cluster_df = append_df.drop(drop_list, 1)
    silhouette_dict = {}
    n_clusters = list(np.arange(2, max_clusters + 1, 1))
    for n in tqdm(n_clusters):
        kmeans = KMeans(n_clusters=n, random_state=19)
        labels = kmeans.fit_predict(cluster_df)
        silhouette_mean = silhouette_score(cluster_df, labels)
        silhouette_dict[n] = silhouette_mean
    best_n = max(silhouette_dict.items(), key=operator.itemgetter(1))[0]
    kmeans = KMeans(n_clusters=best_n, random_state=19)
    labels = kmeans.fit_predict(cluster_df)
    append_df['cluster'] = labels
    return append_df


def get_cluster_summary(df, cluster_column_name, save_path):
    """
    Produces a summary of the cluster results and saves it locally.

    :param df: pandas dataframe produced by run_kmeans_clustering()
    :param cluster_column_name: name of the column that identifies the cluster label
    :param save_path: path in which to save the output
    """
    mean_df = df.groupby(cluster_column_name).mean().reset_index()
    sum_df = df.groupby(cluster_column_name).sum().reset_index()
    count_df = df.groupby(cluster_column_name).count().reset_index()
    mean_df = pd.melt(mean_df, id_vars=[cluster_column_name])
    mean_df.rename(columns={'value': 'mean'}, inplace=True)
    sum_df = pd.melt(sum_df, id_vars=[cluster_column_name])
    sum_df.rename(columns={'value': 'sum'}, inplace=True)
    count_df = pd.melt(count_df, id_vars=[cluster_column_name])
    count_df.rename(columns={'value': 'count'}, inplace=True)
    summary_df = pd.merge(mean_df, sum_df, how='inner', on=['cluster', 'variable'])
    summary_df = pd.merge(summary_df, count_df, how='inner', on=['cluster', 'variable'])
    summary_df.to_csv(os.path.join(save_path, 'cluster_summary.csv'), index=False)


def run_association_rules(df, drop_list, min_support, lift_threshold, save_path):
    """
    Runs association rules mining on the provided data.

    :param df: pandas dataframe we want to run the algorithm on
    :param drop_list: list of features we want to exclude
    :param min_support: the minimum support necessary
    :param lift_threshold: the minimum lift necessary
    :param save_path: path in which to save the output
    :returns: tuple including the 1) pandas dataframe of the association rules meeting min_support and lift_threshold,
    2) pandas dataframe of the transformed df that was used to run association rules mining
    """
    print('running association rules...')
    drop_df = df[drop_list].reset_index(drop=True)
    assoc_df = df.drop(drop_list, 1)
    num_cols = list(assoc_df.select_dtypes(include='number'))

    for col in num_cols:
        assoc_df[col] = pd.qcut(assoc_df[col], q=4, labels=['1', '2', '3', '4']).astype(str)

    assoc_df = pd.get_dummies(assoc_df)
    cat_cols = list(assoc_df)
    for col in cat_cols:
        assoc_df[col] = np.where(assoc_df[col] == 1, True, False)

    frequent_itemsets = fpgrowth(assoc_df, min_support=min_support, use_colnames=True)
    rules_df = association_rules(frequent_itemsets, metric='lift', min_threshold=lift_threshold)
    rules_df.sort_values(by='lift', ascending=False, inplace=True)
    rules_df.to_csv(os.path.join(save_path, 'association_rules.csv'), index=False)
    assoc_df = pd.concat([assoc_df, drop_df], axis=1)
    return rules_df, assoc_df


def get_association_rules_summary(rules_df, assoc_df, interest_column, save_path):
    """
    Summarizes the association rules mining results by the interest_column, which will often be the target.

    :param rules_df: rules_df returned by run_association_rules()
    :param assoc_df: assoc_df returned by run_association_rules()
    :param interest_column: the column we want to summarize by rule
    :param save_path: path in which to save the output
    """
    rules_df['antecedents'] = rules_df['antecedents'].astype(str) + ','
    rules_df['all_rules'] = rules_df['antecedents'] + ' ' + rules_df['consequents'].astype(str)
    rules_df['all_rules'] = rules_df['all_rules'].str.replace('frozenset', '').str.replace('{', '')\
        .str.replace('}', '').str.replace('(', '').str.replace(')', '')
    rules_df['all_rules'] = rules_df['all_rules'].apply(lambda x: x.split(','))

    summary_df = pd.DataFrame()
    for index, row in tqdm(rules_df.iterrows()):
        temp_df = deepcopy(assoc_df)
        for r in row['all_rules']:
            r = r.replace("'", '').lstrip()
            temp_df = temp_df.loc[temp_df[r] == True]
        outcome = temp_df[interest_column].mean()
        temp_summary_df = pd.DataFrame({
            'outcome': [outcome],
            'outcome_count': [len(temp_df)],
            'rules': [row['all_rules']]
        })
        summary_df = summary_df.append(temp_summary_df)
    summary_df.sort_values(by=['outcome'], ascending=False, inplace=True)
    summary_df.to_csv(os.path.join(save_path, 'association_rules_summary.csv'), index=False)


def get_data_to_explore():
    """
    Tightly-coupled function to retrieve the data we want to explore.
    """
    df = pd.read_sql('''select * from churn_model.churn_data;''', make_mysql_connection('churn-model-mysql'))
    df['churn'] = np.where(df['churn'].str.startswith('y'), 1, 0)
    df.drop(['id', 'meta__inserted_at', 'client_id', 'acquired_date'], 1, inplace=True)
    return df


def run_exploration(df, target, files_path, images_path, max_kmeans_clusters, min_assoc_rules_support,
                    assoc_rules_lift_threshold):
    """
    Runs a series of exploration functions, which include producing / finding:
    - TSNE visualization
    - Summary statistics
    _ Highly correlated features
    - Univariate feature importance
    - K-means clustering
    - Association rules mining
    - Categorical dispersion
    - Density plots by target

    :param df: dataframe to explore
    :param target: name of the target column
    :param files_path: path in which to save files
    :param images_path: path in which to save images
    :param max_kmeans_clusters: max number of clusters to consider
    :param min_assoc_rules_support: min support needed for association rules mining
    :param assoc_rules_lift_threshold: min lift needed for association rules mining
    """
    create_tsne_visualization(df, target, images_path)
    generate_summary_statistics(df, target, files_path)
    find_highly_correlated_features(df, files_path)
    score_features_using_mutual_information(df, target, files_path)
    run_predictive_power_score(df, target, files_path)

    cluster_df = run_kmeans_clustering(df, [target], max_kmeans_clusters)
    get_cluster_summary(cluster_df, 'cluster', files_path)
    assoc_rules_df, raw_assoc_df = run_association_rules(df, [target], min_assoc_rules_support,
                                                         assoc_rules_lift_threshold, files_path)
    get_association_rules_summary(assoc_rules_df, raw_assoc_df, target, files_path)

    categorical_cols = list(df.select_dtypes(include='object'))
    numeric_cols = list(df.select_dtypes(include='number').drop([target], 1))
    target_balance_tuple = calculate_binary_target_balance(df, target)
    categorical_dispersion_df = pd.DataFrame()
    category_connection_df = pd.DataFrame()

    for column in categorical_cols:
        temp_categorical_dispersion_df = analyze_categorical_feature_dispersion(df, column)
        categorical_dispersion_df = categorical_dispersion_df.append(temp_categorical_dispersion_df)
        temp_category_connection_df = analyze_category_by_binary_target(df, column, target, target_balance_tuple)
        category_connection_df = category_connection_df.append(temp_category_connection_df)
    categorical_dispersion_df.to_csv(os.path.join(files_path, 'categorical_dispersion.csv'), index=False)
    category_connection_df.to_csv(os.path.join(files_path, 'categorical_connection.csv'), index=False)
    plot_category_level_counts_and_target_connection(category_connection_df, categorical_dispersion_df, images_path)

    for feature in numeric_cols:
        make_density_plot_by_binary_target(df, feature, target, images_path)


def main(target, images_path, files_path, max_kmeans_clusters=7, min_assoc_rules_support=0.1,
         assoc_rules_lift_threshold=2):
    """
    Ingests and explores data.

    :param target: name of the target column
    :param files_path: path in which to save files
    :param images_path: path in which to save images
    :param max_kmeans_clusters: max number of clusters to consider
    :param min_assoc_rules_support: min support needed for association rules mining
    :param assoc_rules_lift_threshold: min lift needed for association rules mining
    """
    start_time = time.time()
    create_exploration_directories(images_path, files_path)
    df = get_data_to_explore()
    run_exploration(
        df=df,
        target=target,
        files_path=files_path,
        images_path=images_path,
        max_kmeans_clusters=max_kmeans_clusters,
        min_assoc_rules_support=min_assoc_rules_support,
        assoc_rules_lift_threshold=assoc_rules_lift_threshold
    )
    print("--- %s seconds for script to run ---" % (time.time() - start_time))


if __name__ == "__main__":
    main(
        target='churn',
        images_path=os.path.join('utilities', 'output_files'),
        files_path=os.path.join('utilities', 'output_images')
    )
