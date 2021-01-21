from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, f1_score, roc_auc_score


DIAGNOSTICS_DIRECTORY = 'diagnostics'
MODELS_DIRECTORY = 'models'
DATA_DIRECTORY = 'data'
TARGET = 'churn'
FEATURES_TO_DROP = ['acquired_date']
TEST_SET_PERCENTAGE = 0.2
CV_SCORER = 'neg_log_loss'
CLASS_CUTOFF = 0.50
CALIBRATION_BINS = 10
S3_BUCKET = 'churn-model-data-science-modeling'
DB_SECRET_NAME = 'churn-model-mysql'
SCHEMA_NAME = 'churn_model'
CV_TIMES = 4
CATEGORICAL_FILL_VALUES = 'missing'


ENGINEERING_PARAM_GRID = {
    'preprocessor__numeric_transformer__log_creator__take_log': ['yes', 'no'],
    'preprocessor__categorical_transformer__category_combiner__combine_categories': ['yes', 'no'],
    'feature_selector__percentile': (1, 100)
}

FOREST_PARAM_GRID = {
    'model__max_depth': (3, 31),
    'model__min_samples_leaf': (0.0001, 0.01),
    'model__max_features': ['log2', 'sqrt']
}

FOREST_CALIBRATED_PARAM_GRID = {
    'model__base_estimator__max_depth': (3, 31),
    'model__base_estimator__min_samples_leaf': (0.0001, 0.01),
    'model__base_estimator__max_features': ['log2', 'sqrt']
}

MODEL_TRAINING_DICT = {
    'extra_trees_uncalibrated': [ExtraTreesClassifier(), FOREST_PARAM_GRID, 20],
    'extra_trees_sigmoid': [CalibratedClassifierCV(ExtraTreesClassifier(), method='sigmoid'),
                            FOREST_CALIBRATED_PARAM_GRID, 20],
    'extra_trees_isotonic': [CalibratedClassifierCV(ExtraTreesClassifier(), method='isotonic'),
                             FOREST_CALIBRATED_PARAM_GRID, 20],
}

MODEL_EVALUATION_LIST = [
    ('1_prob', log_loss, 'log_loss'),
    ('1_prob', roc_auc_score, 'roc_auc'),
    ('predicted_class', f1_score, 'f1'),
]
