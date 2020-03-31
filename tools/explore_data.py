import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings

from functools import partial
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from data.db import connect_to_sqlite_database


warnings.filterwarnings('ignore')


parent_path = Path(os.getcwd()).parent
images_path = 'output_images'
files_path = 'output_files'


def create_exploration_directories():
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    if not os.path.exists(files_path):
        os.makedirs(files_path)


def get_data_for_exploration():
    query = '''
    select 
    loan_status, 
    addr_state, 
    annual_inc, 
    annual_inc_joint, 
    application_type, 
    earliest_cr_line, 
    emp_length, 
    funded_amnt, 
    funded_amnt_inv, 
    grade,
    home_ownership, 
    installment, 
    int_rate, 
    issue_d, 
    loan_amnt, 
    term, 
    verification_status, 
    zip_code,
    disbursement_method
    from loans;'''
    db_conn = connect_to_sqlite_database(os.path.join(parent_path, 'data', 'loans.db'))
    loans_df = pd.read_sql(query, db_conn)
    loans_df.drop(['earliest_cr_line', 'issue_d'], 1, inplace=True)
    loans_df = loans_df.sample(n=100_000, random_state=19)
    return loans_df


def create_scatterplot(df, column1, column2):
    sns.scatterplot(x=column1, y=column2, data=df)
    plt.title('scatterplot for ' + column1 + ' and ' + column2)
    plt.savefig(os.path.join(images_path, 'scatterplot_for_ ' + column1 + '_and_' + column2 + '.png'))
    plt.clf()


def create_histogram(df, column, bins):
    sns.distplot(df[column], bins=bins, kde=False)
    plt.title('histogram for ' + column)
    plt.savefig(os.path.join(images_path, 'histogram_for_' + column + '.png'))
    plt.clf()


def create_density_plot(df, column):
    sns.kdeplot(df[column], shade=True, color='r', legend=True)
    plt.xlabel(column)
    plt.ylabel('density')
    plt.savefig(os.path.join(images_path, 'density_plot_for_' + column + '.png'))
    plt.clf()


def create_tsne_visualization(df, target, target_levels):
    target_0_df = (df.loc[df[target] == target_levels[0]]).head(5_000)
    target_1_df = (df.loc[df[target] == target_levels[1]]).head(5_000)
    tsne_df = pd.concat([target_0_df, target_1_df])

    target_df = tsne_df[[target]]
    tsne_df = tsne_df.select_dtypes(include=['float64', 'float32', 'int'])
    tsne_df.dropna(how='all', inplace=True, axis=1)
    tsne_df = pd.DataFrame(SimpleImputer(strategy='mean', copy=False).fit_transform(tsne_df),
                           columns=list(tsne_df))
    tsne_df = pd.DataFrame(StandardScaler().fit_transform(tsne_df), columns=list(tsne_df))

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(tsne_df)

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
    plt.savefig(os.path.join(images_path, 'tsne.png'))
    plt.clf()


def analyze_categorical_feature_dispersion(df, feature, fill_na_value='unknown'):
    df.fillna({feature: fill_na_value}, inplace=True)
    total_observations = len(df)
    grouped_df = pd.DataFrame(df.groupby(feature)[feature].count())
    grouped_df.columns = ['count']
    grouped_df.reset_index(inplace=True)
    grouped_df['percentage'] = grouped_df['count'] / total_observations

    sns.barplot(x=feature, y='percentage', data=grouped_df)
    plt.title('Categorical Dispersion for ' + feature)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(images_path, 'categorical_dispersion_for_' + feature + '.png'))
    plt.clf()

    grouped_df = grouped_df[[feature, 'count', 'percentage']]
    grouped_df.rename(columns={feature: 'feature'}, inplace=True)
    grouped_df['feature'] = feature + '_' + grouped_df['feature']
    return grouped_df


def make_density_plot_by_binary_target(df, column, target_column, target_levels):
    target_df = df[[target_column]]
    target_df.reset_index(inplace=True, drop=True)
    df = df[[column]]
    df.reset_index(inplace=True, drop=True)
    df = pd.DataFrame(SimpleImputer(strategy='mean', copy=False).fit_transform(df), columns=list(df))
    df = pd.concat([target_df, df], axis=1)
    df_level0 = df.loc[df[target_column] == target_levels[0]]
    df_level0.rename(columns={column: target_levels[0]}, inplace=True)
    p0 = sns.kdeplot(df_level0[target_levels[0]], shade=True, color='r', legend=True)

    df_level1 = df.loc[df[target_column] == target_levels[1]]
    df_level1.rename(columns={column: target_levels[1]}, inplace=True)
    p1 = sns.kdeplot(df_level1[target_levels[1]], shade=True, color='b', legend=True)

    plt.xlabel(column)
    plt.ylabel('density')
    plt.savefig(os.path.join(images_path, 'density_plot_by_target_for_' + column + '.png'))
    plt.clf()


def _calculate_binary_target_balance(df, target):
    total_observations = len(df)
    grouped_df = pd.DataFrame(df.groupby(target)[target].count())
    grouped_df.columns = ['count']
    grouped_df.reset_index(inplace=True)
    grouped_df['percentage'] = grouped_df['count'] / total_observations
    grouped_df = grouped_df[[target, 'percentage']]
    target_balance_tuple = [tuple(x) for x in grouped_df.values]
    return target_balance_tuple


def plot_category_by_binary_target(df, feature, target, target_balance_tuple, fill_na_value='unknown'):
    class_1_name = target_balance_tuple[0][0]
    class_1_percent = target_balance_tuple[0][1]
    class_2_percent = target_balance_tuple[1][1]

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
                                                   grouped_df['feature_count'] * class_2_percent)
    grouped_df['expected_percentage'] = grouped_df['expected_observations'] / grouped_df['feature_count']
    grouped_df['diff_from_expectation'] = abs(grouped_df['actual_percentage'] - grouped_df['expected_percentage'])
    grouped_df = grouped_df.loc[grouped_df[target] == class_1_name]

    sns.barplot(x=feature, y='diff_from_expectation', hue=target, data=grouped_df)
    plt.title('Category Difference from Expectation for ' + feature)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(images_path, 'category_plot_by_target_for_' + feature + '.png'))
    plt.clf()

    grouped_df['feature'] = feature
    grouped_df = grouped_df[['feature', 'count', 'actual_percentage']]
    return grouped_df


def generate_summary_statistics(df, target):
    total_df = df.describe(include='all')
    total_df['level'] = 'total'
    for level in (list(set(df[target].tolist()))):
        temp_df = df.loc[df[target] == level]
        temp_df = temp_df.describe(include='all')
        temp_df['level'] = level
        total_df = total_df.append(temp_df)
    total_df.to_csv(os.path.join(files_path, 'summary_statistics.csv'), index=True)


def find_highly_correlated_features(df, correlation_cutoff):
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
    df_high = df.loc[df['correlation'] >= correlation_cutoff]
    df_high.to_csv(os.path.join(files_path, 'highly_correlated_features.csv'), index=False)


def find_best_features_using_mutual_information(df, target, feature_percentage):
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
    feature_scores = feature_scores[0: int(len(feature_scores) * feature_percentage)]
    feature_scores.to_csv(os.path.join(files_path, 'top_univariate_features_mi.csv'), index=False)


if __name__ == "__main__":
    start_time = time.time()
    create_exploration_directories()
    loans_df = get_data_for_exploration()

    create_tsne_visualization(loans_df, 'loan_status', ['Fully Paid', 'Charged Off'])
    generate_summary_statistics(loans_df, 'loan_status')
    find_highly_correlated_features(loans_df, 0.99)
    find_best_features_using_mutual_information(loans_df, 'loan_status', 0.2)

    categorical_df = loans_df.select_dtypes(include='object').drop(['loan_status'], 1)
    numeric_df = loans_df.select_dtypes(include='number')

    target_balance_tuple = _calculate_binary_target_balance(loans_df, 'loan_status')
    categorical_dispersion_df = pd.DataFrame()
    start_time_of_loops = time.time()
    for column in list(categorical_df):
        temp_categorical_disperson_df = analyze_categorical_feature_dispersion(loans_df, column)
        categorical_dispersion_df = categorical_dispersion_df.append(temp_categorical_disperson_df)
        plot_category_by_binary_target(loans_df, column, 'loan_status', target_balance_tuple)
    categorical_dispersion_df.to_csv(os.path.join(files_path, 'categorical_dispersion.csv'), index=False)

    for column in list(numeric_df):
        make_density_plot_by_binary_target(loans_df, column, 'loan_status', ['Fully Paid', 'Charged Off'])
    print("--- %s seconds for the for loop---" % (time.time() - start_time_of_loops))
    print("--- %s seconds for script to run ---" % (time.time() - start_time))
