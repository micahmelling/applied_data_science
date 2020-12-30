import os
import pandas as pd
import numpy as np
import pytz
import warnings

from ds_helpers import aws
from datetime import datetime
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from modeling.config import S3_BUCKET

warnings.filterwarnings('ignore')


def upload_model_directory_to_s3(model_directory):
    """
    Uploads an entire model's directory of data and diagnostics, along with the model itself, to S3.

    :param model_directory: name of the model's directory
    """
    print(f'uploading all files in {model_directory}')
    directory_walk = os.walk(model_directory)
    for directory_path, directory_name, file_names in directory_walk:
        if directory_path != os.path.join(model_directory):
            sub_dir = os.path.basename(directory_path)
            for file in tqdm(file_names):
                aws.upload_file_to_s3(file_name=os.path.join(model_directory, sub_dir, file), bucket=S3_BUCKET)


def make_directories_if_not_exists(directories_list):
    """
    Makes directories in the current working directory if they do not exist:

    :param directories_list: list of directories to create
    """
    for directory in directories_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


def create_model_uid(model_name):
    """
    Creates a UID for a model.

    :param model_name: the base name the model (e.g. random_forest)
    :returns: unique string
    """
    tz = pytz.timezone('US/Central')
    now = str(datetime.now(tz))
    now = now.replace(' ', '').replace(':', '').replace('.', '').replace('-', '')
    model_uid = model_name + '_' + now
    return model_uid


def fill_missing_values(df, fill_value):
    """
    Fills all missing values in a dataframe with fill_value.

    :param df: pandas dataframe
    :param fill_value: the fill value
    :returns: pandas dataframe
    """
    df = df.fillna(value=fill_value)
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


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Replaces a categorical level with its mean value.
    """

    def __init__(self, sparsity_cutoff=0.001):
        self.mapping_dict = {}
        self.sparsity_cutoff = sparsity_cutoff

    def fit(self, X, Y):
        target_name = list(pd.DataFrame(Y))
        xy_df = pd.concat([Y, X], axis=1)
        column_names = list(xy_df.drop(target_name, 1))
        overall_mean = Y.mean()

        for column in column_names:
            col_dict = {}
            uniques = xy_df[column].unique()
            for unique in uniques:
                percentage = xy_df[xy_df[column] == unique].count()[0] / len(xy_df)
                if percentage > self.sparsity_cutoff:
                    col_dict[unique] = xy_df[xy_df[column] == unique].mean()[0]
                else:
                    col_dict[unique] = overall_mean
            self.mapping_dict[column] = col_dict

        return self

    def transform(self, X, Y=None):
        for key, value in self.mapping_dict.items():
            X[key] = X[key].map(value)
        return X
