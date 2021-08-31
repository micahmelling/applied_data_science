import os
import pandas as pd
import numpy as np
import pytz
import warnings
import joblib

from copy import deepcopy
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


warnings.filterwarnings('ignore')


def save_data_in_model_directory(model_uid, x_train, x_test, y_train, y_test):
    """
    Saves training data into the model directory.

    :param model_uid: model uid
    :param x_train: x train
    :param x_test: x test
    :param y_train: y train
    :param y_test: y test
    """
    model_uid_data_directory = os.path.join('modeling', model_uid, 'data')
    make_directories_if_not_exists([model_uid_data_directory])
    joblib.dump(x_train, os.path.join(model_uid_data_directory, 'x_train.pkl'), compress=9)
    joblib.dump(x_test, os.path.join(model_uid_data_directory, 'x_test.pkl'), compress=9)
    joblib.dump(y_train, os.path.join(model_uid_data_directory, 'y_train.pkl'), compress=9)
    joblib.dump(y_test, os.path.join(model_uid_data_directory, 'y_test.pkl'), compress=9)


def ensure_features_are_standardized(df, feature_mapping):
    """
    Ensures df 1) includes only the features in feature_mapping and 2) adheres to the dtype mappings in feature_mapping.

    :param df: pandas dataframe
    :param feature_mapping: dictionary where keys are column names and values are dtypes
    """
    df = df.drop(columns=[col for col in df if col not in feature_mapping])
    cols = list(df)
    for key, value in feature_mapping.items():
        if key not in cols:
            df[key] = np.nan
    df = df.astype(feature_mapping)
    df = df[list(feature_mapping.keys())]
    return df


def find_non_dummied_columns(df):
    """
    Finds the names of columns that have not been dummied.

    :param df: pandas dataframe
    :returns: list
    """
    cols = list(df)
    non_dummied_cols = []
    for col in cols:
        unique_col_vals = list(df[col].unique())
        unique_col_vals = [int(v) for v in unique_col_vals]
        if set(unique_col_vals) not in [{0, 1}, {0}, {1}]:
            non_dummied_cols.append(col)
    return non_dummied_cols


def make_directories_if_not_exists(directories_list):
    """
    Makes directories in the current working directory if they do not exist:

    :param directories_list: list of directories to create
    """
    for directory in directories_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


def create_uid(base_string):
    """
    Creates a UID by concatenating the current timestamp to base_string.

    :param base_string: the base string
    :returns: unique string
    """
    tz = pytz.timezone('US/Central')
    now = str(datetime.now(tz))
    now = now.replace(' ', '').replace(':', '').replace('.', '').replace('-', '')
    uid = base_string + '_' + now
    return uid


def save_pipeline(pipeline, model_uid, subdirectory):
    """
    Saves a modeling pipeline locally as a pkl file into the model_uid's directory.

    :param pipeline: scikit-learn pipeline
    :param model_uid: model uid
    :param subdirectory: subdirectory in which to save the pipeline
    """
    save_directory = os.path.join('modeling', model_uid, subdirectory)
    make_directories_if_not_exists([save_directory])
    joblib.dump(pipeline, os.path.join(save_directory, 'model.pkl'), compress=3)


def save_cv_scores(df, model_uid, subdirectory):
    """
    Saves cross validation scores locally as a csv file into the model_uid's directory.

    :param df: dataframe of cv scores
    :param model_uid: model uid
    :param subdirectory: subdirectory in which to save the output
    """
    save_directory = os.path.join('modeling', model_uid, subdirectory)
    make_directories_if_not_exists([save_directory])
    df.to_csv(os.path.join(save_directory, 'cv_scores.csv'), index=False)


def determine_if_name_in_object(name, py_object):
    """
    Determine if a name is in a Python object.

    :param py_object: Python object
    :param name: name to search for in py_object
    """
    object_str = str((type(py_object))).lower()
    if name in object_str:
        return True
    else:
        return False


def fill_missing_values(df, fill_value):
    """
    Fills all missing values in a dataframe with fill_value.

    :param df: pandas dataframe
    :param fill_value: the fill value
    :returns: pandas dataframe
    """
    df = df.fillna(value=fill_value)
    df = df.replace('nan', fill_value)
    return df


def clip_feature_bounds(df, feature, cutoff, new_amount, clip_type):
    """
    Re-assigns values above or below a certain threshold.

    :param df: pandas dataframe
    :param feature: name of the feature to clip
    :param cutoff: the point beyond which the value is changed
    :param new_amount: the amount to assign points beyond cutoff
    :param clip_type: denotes if we want to change values above or below the cutoff; can either be upper or lower
    :returns: pandas dataframe
    """
    if clip_type == 'upper':
        df[feature] = np.where(df[feature] > cutoff, new_amount, df[feature])
    elif clip_type == 'lower':
        df[feature] = np.where(df[feature] < cutoff, new_amount, df[feature])
    else:
        raise Exception('clip_type must either be upper or lower')
    return df


def drop_features(df, feature_list):
    """
    Drops features from a dataframe.

    :param df: pandas dataframe
    :param feature_list: list of features to drop
    :returns: pandas dataframe
    """
    df = df.drop(feature_list, 1)
    return df


def map_dict_to_column(df, feature, mapping_dict):
    """
    Updates a column's values per the passed mapping_dict.

    :param df: pandas dataframe
    :param feature: name of the feature to map
    :param mapping_dict: dictionary to map onto feature
    :returns: pandas dataframe
    """
    df[feature] = df[feature].map(mapping_dict)
    return df


def convert_column_to_datetime(df, feature):
    """
    Converts a column to datetime data type.

    :param df: pandas dataframe
    :param feature: name of the feature to convert
    :returns: pandas dataframe
    """
    df[feature] = pd.to_datetime(df[feature])
    return df


def extract_month_from_date(df, date_col):
    """
    Creates a month column, called month, from an existing date.

    :param df: pandas dataframe
    :param date_col: name of the date column; expected to be of time pandas datetime
    :returns: pandas dataframe
    """
    df['month'] = df[date_col].dt.month.astype(str)
    return df


def convert_month_to_quarter(df, month_col, mapping_dict):
    """
    Wrapper function to convert a month column into a quarter.

    :param df: pandas dataframe
    :param month_col: name of the month column
    :param mapping_dict: dictionary that maps a month into a quarter
    :returns: pandas dataframe
    """
    df = map_dict_to_column(df, month_col, mapping_dict)
    df = df.rename(columns={month_col: 'quarter'})
    return df


def extract_year_from_date(df, date_col):
    """
    Creates a year column, called year, from an existing date.

    :param df: pandas dataframe
    :param date_col: name of the date column; expected to be of time pandas datetime
    :returns: pandas dataframe
    """
    df['year'] = df[date_col].dt.year.astype(str)
    return df


def create_ratio_column(df, col1, col2):
    """
    Creates a column that is a ratio of col1 and col2.

    :param df: pandas dataframe
    :param col1: name of first column
    :param col2: name of second column
    """
    df[col1 + '_' + col2 + '_ratio'] = df[col1] / df[col2]
    return df


def create_x_y_split(df, target):
    """
    Splits a dataframe into predictor features (x) and the target values (y).

    :param df: pandas dataframe
    :param target: name of the target feature
    :returns: x dataframe of predictors, y series of the target
    """
    y = df[target]
    x = df.drop(target, 1)
    return x, y


def create_train_test_split(x, y, test_size=0.25):
    """
    Creates a train-test split for training and evaluation of machine learning models.

    :param x: dataframe of predictor features
    :param y: series of target values
    :param test_size: percentage of observations to assign to the test set
    :returns: x_train, x_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=19)
    return x_train, x_test, y_train, y_test


class TakeLog(BaseEstimator, TransformerMixin):
    """
    Based on the argument, takes the log of the numeric columns.
    """

    def __init__(self, take_log='yes'):
        self.take_log = take_log

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        if self.take_log == 'yes':
            for col in list(X):
                X[col] = np.log(X[col])
                X[col] = X[col].replace([np.inf, -np.inf], 0)
                return X
        elif self.take_log == 'no':
            return X
        else:
            return X


class CombineCategoryLevels(BaseEstimator, TransformerMixin):
    """
    Combines category levels that individually fall below a certain percentage of the total.
    """
    def __init__(self, combine_categories='yes', sparsity_cutoff=0.001):
        self.combine_categories = combine_categories
        self.sparsity_cutoff = sparsity_cutoff
        self.mapping_dict = {}

    def fit(self, X, Y=None):
        for col in list(X):
            percentages = X[col].value_counts(normalize=True)
            combine = percentages.loc[percentages <= self.sparsity_cutoff]
            combine_levels = combine.index.tolist()
            self.mapping_dict[col] = combine_levels
        return self

    def transform(self, X, Y=None):
        if self.combine_categories == 'yes':
            for col in list(X):
                combine_cols = self.mapping_dict.get(col, [None])
                X.loc[X[col].isin(combine_cols), col] = 'sparse_combined'
            return X
        elif self.combine_categories == 'no':
            return X
        else:
            return X


class FeaturesToDict(BaseEstimator, TransformerMixin):
    """
    Converts dataframe, or numpy array, into a dictionary oriented by records. This is a necessary pre-processing step
    for DictVectorizer().
    """
    def __int__(self):
        pass

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = X.to_dict(orient='records')
        return X


def transform_data_with_pipeline(pipeline, x_df):
    """
    Prepares the model and x_test dataframe for extracting feature importance values. This involves applying the
    preprocessing stepsin the pipeline and converting the output into a dataframe with the appropriate columns.
    Likewise, this process involves plucking out the model.

    :param pipeline: scikit-learn pipeline with preprocessing steps and model
    :param x_df: x_test dataframe
    computationally expensive; default is 10,000. If n_obs is greater than the total number of observations, then
    50% of the data will be sampled.
    :returns: model with predict method, transformed x_test dataframe
    """
    # make a copy of the pipeline to avoid implications of in-place operations
    pipeline_ = deepcopy(pipeline)

    # Extract the names of the features from the dict vectorizers
    num_dict_vect = pipeline_.named_steps['preprocessor'].named_transformers_.get('numeric_transformer').named_steps[
        'dict_vectorizer']
    cat_dict_vect = pipeline_.named_steps['preprocessor'].named_transformers_.get('categorical_transformer').named_steps[
        'dict_vectorizer']
    num_features = num_dict_vect.feature_names_
    cat_features = cat_dict_vect.feature_names_

    # Get the boolean masks for the variance threshold and feature selector steps
    num_feature_selector_support = pipeline_.named_steps['preprocessor'].named_transformers_.get(
        'numeric_transformer').named_steps['feature_selector'].get_support()
    cat_feature_selector_support = pipeline_.named_steps['preprocessor'].named_transformers_.get(
        'categorical_transformer').named_steps['feature_selector'].get_support()
    variance_threshold_support = pipeline_.named_steps['variance_thresholder'].get_support()

    # Create a dataframe of column names
    cols_df = pd.DataFrame({'cols': num_features + cat_features})

    # Remove columns based on the feature selectors
    cols_df = pd.concat([
        cols_df,
        pd.DataFrame({'selector_support': list(num_feature_selector_support) + list(cat_feature_selector_support)})
    ], axis=1)
    cols_df = cols_df.loc[cols_df['selector_support']]
    cols_df = cols_df.reset_index()

    # Remove columns based on the variance threshold
    cols_df = pd.concat([
        cols_df,
        pd.DataFrame({'threshold_support': variance_threshold_support})
    ], axis=1)
    cols_df = cols_df.loc[cols_df['threshold_support']]

    # Make list of final column names
    cols = cols_df['cols'].tolist()

    # Remove the model
    pipeline_.steps.pop(len(pipeline_) - 1)

    # Transform the data using the remaining pipeline_ steps, cast to a dataframe, and assign the column names
    x_df = pipeline_.transform(x_df)
    x_df = pd.DataFrame(x_df)
    x_df.columns = cols

    return x_df
