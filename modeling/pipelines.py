from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectPercentile, f_classif, VarianceThreshold
from sklearn.feature_extraction import DictVectorizer

from helpers.model_helpers import clip_feature_bounds, drop_features, convert_column_to_datetime, \
    extract_month_from_date, convert_month_to_quarter, extract_year_from_date, create_ratio_column, TakeLog, \
    CombineCategoryLevels, FeaturesToDict
from helpers.constants import MONTH_TO_QUARTER_DICT
from modeling.config import FEATURES_TO_DROP


def get_pipeline(model):
    numeric_transformer = Pipeline(steps=[
        ('mouse_movement_clipper', FunctionTransformer(clip_feature_bounds, validate=False,
                                                       kw_args={'feature': 'mouse_movement', 'cutoff': 0,
                                                                'new_amount': 0, 'clip_type': 'lower'})),
        ('propensity_score_clipper', FunctionTransformer(clip_feature_bounds, validate=False,
                                                         kw_args={'feature': 'propensity_score', 'cutoff': 0,
                                                                  'new_amount': 0, 'clip_type': 'lower'})),
        ('completeness_score_clipper', FunctionTransformer(clip_feature_bounds, validate=False,
                                                           kw_args={'feature': 'completeness_score', 'cutoff': 0,
                                                                    'new_amount': 0, 'clip_type': 'lower'})),
        ('profile_score_clipper', FunctionTransformer(clip_feature_bounds, validate=False,
                                                      kw_args={'feature': 'profile_score', 'cutoff': 0,
                                                               'new_amount': 0, 'clip_type': 'lower'})),
        ('ratio_creator', FunctionTransformer(create_ratio_column, validate=False,
                                              kw_args={'col1': 'profile_score', 'col2': 'activity_score'})),
        ('log_creator', TakeLog()),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('date_transformer', FunctionTransformer(convert_column_to_datetime, validate=False,
                                                 kw_args={'feature': 'acquired_date'})),
        ('month_extractor', FunctionTransformer(extract_month_from_date, validate=False,
                                                kw_args={'date_col': 'acquired_date'})),
        ('quarter_extractor', FunctionTransformer(convert_month_to_quarter, validate=False,
                                                  kw_args={'month_col': 'month',
                                                           'mapping_dict': MONTH_TO_QUARTER_DICT})),
        ('year_extractor', FunctionTransformer(extract_year_from_date, validate=False,
                                               kw_args={'date_col': 'acquired_date'})),
        ('category_combiner', CombineCategoryLevels()),
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown', add_indicator=True)),
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer()),
        ('variance_thresholder', VarianceThreshold(threshold=(0.999 * (1 - 0.999))))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_transformer', numeric_transformer, selector(dtype_exclude='category')),
            ('categorical_transformer', categorical_transformer, selector(dtype_include='category'))
        ])

    pipeline = Pipeline(steps=[
        ('feature_dropper', FunctionTransformer(drop_features, validate=False,
                                                kw_args={'features_drop_list': FEATURES_TO_DROP})),
        ('preprocessor', preprocessor),
        ('feature_selector', SelectPercentile(f_classif)),
        ('model', model)
    ])

    return pipeline
