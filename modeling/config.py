from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, f1_score, roc_auc_score, brier_score_loss, balanced_accuracy_score
from scipy.stats import randint, uniform

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

from helpers.custom_scorers import calculate_log_loss_brier_score_mix


DIAGNOSTICS_DIRECTORY = 'diagnostics'
MODELS_DIRECTORY = 'models'
DATA_DIRECTORY = 'data'
TARGET = 'churn'
FEATURES_TO_DROP = ['acquired_date']
TEST_SET_PERCENTAGE = 0.2
CV_SCORER = 'neg_log_loss'
CLASS_CUTOFF = 0.50
CALIBRATION_BINS = [10, 25, 50]
S3_BUCKET = 'churn-model-data-science-modeling'
DB_SECRET_NAME = 'churn-model-mysql'
SCHEMA_NAME = 'churn_model'
CV_TIMES = 4
CATEGORICAL_FILL_VALUES = 'missing'


ENGINEERING_PARAM_GRID = {
    'preprocessor__numeric_transformer__log_creator__take_log': ['yes', 'no'],
    'preprocessor__categorical_transformer__category_combiner__combine_categories': ['yes', 'no'],
    'feature_selector__percentile': randint(1, 100)
}

# ENGINEERING_PARAM_GRID = {
#     'preprocessor__numeric_transformer__log_creator__take_log': ['yes'],
#     'preprocessor__categorical_transformer__category_combiner__combine_categories': ['yes'],
#     'feature_selector__percentile': [100]
# }

FOREST_PARAM_GRID = {
    'model__max_depth': randint(3, 31),
    'model__min_samples_leaf': uniform(0.0001, 0.01),
    'model__max_features': ['log2', 'sqrt']
}

FOREST_CALIBRATED_PARAM_GRID = {
    'model__base_estimator__max_depth': (3, 31),
    'model__base_estimator__min_samples_leaf': (0.0001, 0.01),
    'model__base_estimator__max_features': ['log2', 'sqrt']
}

HEURISTIC_MODEL_PARAM_GRID = {
    'feature_selector__percentile': [100],
    'model__maximum': randint(10, 1_000)
}

DUMMY_MODEL_PARAM_GRID = {
}

LOG_REG_PARAM_GRID = {
    'model__C': uniform(0.001, 10)
}


TREE_HDL = {
    "preprocessor__numeric_transformer__log_creator__take_log": {"_type": "choice", "_value": ["yes", "no"],
                                                                 "_default": "yes"},
    "preprocessor__categorical_transformer__category_combiner__combine_categories": {"_type": "choice",
                                                                                     "_value": ["yes", "no"],
                                                                                     "_default": "yes"},
    "feature_selector__percentile": {"_type": "int_uniform", "_value": [1, 100], "_default": 50},
    "model__max_features": {"_type": "choice", "_value": ["sqrt", "log2"], "_default": "sqrt"},
    "model__min_samples_leaf": {"_type": "uniform", "_value": [0.0001, 0.01], "_default": 0.001},
    "model__max_depth": {"_type": "uniform", "_value": [3, 31], "_default": 3}
}


tree_cs = ConfigurationSpace()
take_log = CategoricalHyperparameter("preprocessor__numeric_transformer__log_creator__take_log", ["yes", "no"])
combine_categories = \
    CategoricalHyperparameter("preprocessor__categorical_transformer__category_combiner__combine_categories" ,
                              ["yes", "no"])
feature_selector = UniformIntegerHyperparameter("feature_selector__percentile", 1, 100, default_value=50)
max_features = CategoricalHyperparameter("model__max_features", ["sqrt", "log2"])
min_samples_in_leaf = UniformFloatHyperparameter("model__min_samples_leaf", 0.0001, 0.01, default_value=0.001)
max_depth = UniformIntegerHyperparameter("model__max_depth", 3, 31, default_value=10)
tree_cs.add_hyperparameters([take_log, combine_categories, feature_selector, max_features, min_samples_in_leaf,
                             max_depth])


MODEL_TRAINING_DICT = {
    # 'logistic_regression': [LogisticRegression(solver='sag'), LOG_REG_PARAM_GRID, 10],
    # 'extra_trees_uncalibrated': [ExtraTreesClassifier(), FOREST_PARAM_GRID, 1],
    'extra_trees': [ExtraTreesClassifier(), tree_cs, 4],
    # 'extra_trees_sigmoid': [CalibratedClassifierCV(ExtraTreesClassifier(), method='sigmoid'),
    #                         FOREST_CALIBRATED_PARAM_GRID, 20],
    # 'extra_trees_isotonic': [CalibratedClassifierCV(ExtraTreesClassifier(), method='isotonic'),
    #                          FOREST_CALIBRATED_PARAM_GRID, 20],
}

MODEL_EVALUATION_LIST = [
    ('1_prob', log_loss, 'log_loss'),
    ('1_prob', brier_score_loss, 'brier_score'),
    ('1_prob', roc_auc_score, 'roc_auc'),
    ('predicted_class', f1_score, 'f1'),
    ('predicted_class', balanced_accuracy_score, 'balanced_accuracy'),
    ('1_prob', calculate_log_loss_brier_score_mix, 'log_loss_brier_score_mix')
]
