from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import uniform, randint
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
    'feature_selector__percentile': randint(1, 101)
}


LOG_REG_PARAM_GRID = {
    'model__C': uniform(scale=10),
}


FOREST_PARAM_GRID = {
    'model__max_depth': randint(3, 31),
    'model__min_samples_leaf': uniform(0.0001, 0.1),
    'model__max_features': ['log2', 'sqrt'],
}


XGBoost_PARAM_GRID = {
    'model__base_estimator__learning_rate': uniform(0.01, 0.5),
    'model__base_estimator__n_estimators': randint(50, 200),
    'model__base_estimator__max_depth': randint(3, 31),
    'model__base_estimator__min_child_weight': randint(1, 16),
}


CATBOOST_PARAM_GRID = {
    'model__base_estimator__learning_rate': uniform(0.01, 0.5),
    'model__base_estimator__l2_leaf_reg': randint(1, 11),
    'model__base_estimator__depth': randint(3, 31),
    'model__base_estimator__min_data_in_leaf': randint(1, 16),
}


LIGHTGBM_PARAM_GRID = {
    'model__base_estimator__learning_rate': uniform(0.01, 0.5),
    'model__base_estimator__n_estimators': randint(50, 200),
    'model__base_estimator__max_depth': randint(3, 31),
    'model__base_estimator__min_data_in_leaf': randint(1, 101),
    'model__base_estimator__num_leaves': randint(10, 101),
}


VOTING_PARAM_GRID = {
    'model__base_estimator__log_reg__C': uniform(scale=10),
    'model__base_estimator__random_forest__max_depth': randint(3, 31),
    'model__base_estimator__random_forest__min_samples_leaf': uniform(0.0001, 0.1),
    'model__base_estimator__random_forest__max_features': ['log2', 'sqrt'],
    'model__base_estimator__xgboost__learning_rate': uniform(0.01, 0.5),
    'model__base_estimator__xgboost__n_estimators': randint(50, 200),
    'model__base_estimator__xgboost__max_depth': randint(3, 31),
    'model__base_estimator__xgboost__min_child_weight': randint(1, 16),
}


STACKING_PARAM_GRID = {
    'model__base_estimator__random_forest__max_depth': randint(3, 31),
    'model__base_estimator__random_forest__min_samples_leaf': uniform(0.0001, 0.1),
    'model__base_estimator__random_forest__max_features': ['log2', 'sqrt'],
    'model__base_estimator__light_gbm__learning_rate': uniform(0.01, 0.5),
    'model__base_estimator__light_gbm__n_estimators': randint(50, 200),
    'model__base_estimator__light_gbm__max_depth': randint(3, 31),
    'model__base_estimator__light_gbm__min_data_in_leaf': randint(1, 101),
    'model__base_estimator__light_gbm__num_leaves': randint(10, 101),
}


MODEL_TRAINING_DICT = {
    'logistic_regression': [LogisticRegression(), LOG_REG_PARAM_GRID, 100],
    'random_forest': [CalibratedClassifierCV(RandomForestClassifier()), FOREST_PARAM_GRID, 10],
    'extra_trees': [CalibratedClassifierCV(base_estimator=ExtraTreesClassifier(n_estimators=500)), FOREST_PARAM_GRID,
                    50],
    'xgboost': [CalibratedClassifierCV(base_estimator=XGBClassifier(), cv=4, n_jobs=-1), XGBoost_PARAM_GRID, 25],
    'lightgbm': [CalibratedClassifierCV(base_estimator=LGBMClassifier(verbose=False), cv=4, n_jobs=-1),
                 LIGHTGBM_PARAM_GRID, 50],
    'catboost': [CalibratedClassifierCV(base_estimator=CatBoostClassifier(silent=True, n_estimators=250), cv=4,
                                        n_jobs=-1), CATBOOST_PARAM_GRID, 10],
    'voting_classifier': [CalibratedClassifierCV(base_estimator=VotingClassifier(estimators=[
            ('log_reg', LogisticRegression(solver='sag')),
            ('random_forest', RandomForestClassifier()),
            ('xgboost', XGBClassifier())
        ], voting='soft', n_jobs=-1), cv=4, n_jobs=-1), VOTING_PARAM_GRID, 25],
    'stacking_classifier': [CalibratedClassifierCV(
        base_estimator=StackingClassifier(estimators=[
            ('random_forest', RandomForestClassifier()),
            ('light_gbm', LGBMClassifier(verbose=False))
        ], cv=4, n_jobs=-1), cv=4, n_jobs=-1), STACKING_PARAM_GRID, 25],
}


MODEL_EVALUATION_LIST = [
    ('1_prob', log_loss, 'log_loss'),
    ('1_prob', roc_auc_score, 'roc_auc'),
    ('predicted_class', f1_score, 'f1'),
]
