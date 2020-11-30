from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import uniform, randint
from sklearn.metrics import log_loss, f1_score, roc_auc_score


DIAGNOSTICS_DIRECTORY = 'diagnostics'
MODELS_DIRECTORY = 'models'
DATA_DIRECTORY = '../data/training_data'
TARGET = 'churn'
FEATURES_TO_DROP = ['client_id', 'acquired_date']
TEST_SET_PERCENTAGE = 0.2
CV_SCORER = 'neg_log_loss'


FOREST_PARAM_GRID = {
    'model__base_estimator__max_depth': randint(3, 16),
    'model__base_estimator__min_samples_leaf': randint(1, 10),
    'model__base_estimator__max_features': ['log2', 'sqrt'],
}


XGBoost_PARAM_GRID = {
    'model__base_estimator__learning_rate': uniform(0.01, 0.5),
    'model__base_estimator__n_estimators': randint(50, 200),
    'model__base_estimator__max_depth': randint(3, 16),
    'model__base_estimator__min_child_weight': randint(1, 10),
}


CATBOOST_PARAM_GRID = {
    'model__base_estimator__learning_rate': uniform(0.01, 0.5),
    'model__base_estimator__l2_leaf_reg': randint(1, 10),
    'model__base_estimator__depth': randint(3, 16),
    'model__base_estimator__min_data_in_leaf': randint(1, 10),
}


LIGHTGBM_PARAM_GRID = {
    'model__base_estimator__learning_rate': uniform(0.01, 0.5),
    'model__base_estimator__n_estimators': randint(50, 200),
    'model__base_estimator__max_depth': randint(3, 16),
    'model__base_estimator__min_data_in_leaf': randint(1, 10),
}


MODEL_TRAINING_DICT = {
    'random_forest': [CalibratedClassifierCV(base_estimator=RandomForestClassifier(n_estimators=500)),
                      FOREST_PARAM_GRID, 100],
    'extra_trees': [CalibratedClassifierCV(base_estimator=ExtraTreesClassifier(n_estimators=500)), FOREST_PARAM_GRID,
                    100],
    'xgboost': [CalibratedClassifierCV(base_estimator=XGBClassifier()), XGBoost_PARAM_GRID, 100],
    'lightgbm': [CalibratedClassifierCV(base_estimator=LGBMClassifier()), LIGHTGBM_PARAM_GRID, 100],
    'catboost': [CalibratedClassifierCV(base_estimator=CatBoostClassifier()), CATBOOST_PARAM_GRID, 100]
}


MODEL_EVALUATION_LIST = [
    ('1_prob', log_loss, 'log_loss'),
    ('1_prob', roc_auc_score, 'roc_auc'),
    ('predicted_class', f1_score, 'f1'),
]
