import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


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
    df['month'] = df[date_col].dt.month
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
    df['month'] = df[date_col].dt.year
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
