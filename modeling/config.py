from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, f1_score, roc_auc_score, brier_score_loss, balanced_accuracy_score
from collections import namedtuple
from hyperopt import hp


TARGET = 'churn'
FEATURES_TO_DROP = ['acquired_date']
TEST_SET_PERCENTAGE = 0.20
CLASS_CUTOFF = 0.50
CV_SCORER = 'neg_log_loss'
CV_FOLDS = 5
CATEGORICAL_FILL_VALUE = 'missing'
CALIBRATION_PLOT_BINS = [10, 25, 50, 100]
LOGGING_S3_BUCKET = 'churn-model-data-science-modeling'
DB_SECRET_NAME = 'churn-model-mysql'
DB_SCHEMA_NAME = 'churn_model'
LOG_TO_DB = True
DROP_COL_SCORER = log_loss
DROP_COL_SCORER_STRING = 'neg_log_loss'
DROP_COL_SCORING_TYPE = 'probability'
DROP_COL_HIGHER_IS_BETTER = True
EXPLANATION_SAMPLE_N = 10_000
USE_SHAP_KERNEL = False


FEATURE_DTYPE_MAPPING = {
    'activity_score': float,
    'propensity_score': float,
    'profile_score_new': float,
    'completeness_score': float,
    'xp_points': float,
    'profile_score': float,
    'portfolio_score': float,
    'mouse_movement': float,
    'average_stars': float,
    'ad_target_group': str,
    'marketing_message': str,
    'device_type': str,
    'all_star_group': str,
    'mouse_x': str,
    'coupon_code': str,
    'ad_engagement_group': str,
    'user_group': str,
    'browser_type': str,
    'email_code': str,
    'marketing_creative': str,
    'secondary_user_group': str,
    'promotion_category': str,
    'marketing_campaign': str,
    'mouse_y': str,
    'marketing_channel': str,
    'marketing_creative_sub': str,
    'site_level': str,
    'acquired_date': str
}


ENGINEERING_PARAM_GRID = {
    'preprocessor__numeric_transformer__log_creator__take_log': hp.choice(
        'preprocessor__numeric_transformer__log_creator__take_log', ['yes', 'no']),
    'preprocessor__categorical_transformer__category_combiner__combine_categories': hp.choice(
        'preprocessor__categorical_transformer__category_combiner__combine_categories', ['yes', 'no']),
    'preprocessor__categorical_transformer__feature_selector__percentile': hp.uniformint(
        'preprocessor__categorical_transformer__feature_selector__percentile', 1, 100),
    'preprocessor__numeric_transformer__feature_selector__percentile': hp.uniformint(
        'preprocessor__numeric_transformer__feature_selector__percentile', 1, 100),
}

FOREST_PARAM_GRID = {
    'model__base_estimator__max_depth': hp.uniformint('model__base_estimator__max_depth', 3, 16),
    'model__base_estimator__min_samples_leaf': hp.uniform('model__base_estimator__min_samples_leaf', 0.001, 0.01),
    'model__base_estimator__max_features': hp.choice('model__base_estimator__max_features', ['log2', 'sqrt']),
}

XGBOOST_PARAM_GRID = {
    'model__base_estimator__learning_rate': hp.uniform('model__base_estimator__learning_ratee', 0.01, 0.5),
    'model__base_estimator__n_estimators': hp.randint('model__base_estimator__n_estimators', 75, 150),
    'model__base_estimator__max_depth': hp.randint('model__base_estimator__max_depth', 3, 16),
    'model__base_estimator__min_child_weight': hp.uniformint('model__base_estimator__min_child_weight', 2, 16),
}


model_named_tuple = namedtuple('model_config', {'model_name', 'model', 'param_space', 'iterations'})
MODEL_TRAINING_LIST = [
    model_named_tuple(model_name='random_forest', model=CalibratedClassifierCV(base_estimator=RandomForestClassifier()),
                      param_space=FOREST_PARAM_GRID, iterations=15),
    model_named_tuple(model_name='extra_trees', model=CalibratedClassifierCV(base_estimator=ExtraTreesClassifier()),
                      param_space=FOREST_PARAM_GRID, iterations=15),
    model_named_tuple(model_name='xgboost', model=CalibratedClassifierCV(base_estimator=XGBClassifier()),
                      param_space=XGBOOST_PARAM_GRID, iterations=15),
]


evaluation_named_tuple = namedtuple('model_evaluation', {'evaluation_column', 'scorer_callable', 'metric_name'})
MODEL_EVALUATION_LIST = [
    evaluation_named_tuple(evaluation_column='1_prob', scorer_callable=log_loss, metric_name='log_loss'),
    evaluation_named_tuple(evaluation_column='1_prob', scorer_callable=brier_score_loss,
                           metric_name='brier_score_loss'),
    evaluation_named_tuple(evaluation_column='1_prob', scorer_callable=roc_auc_score, metric_name='roc_auc'),
    evaluation_named_tuple(evaluation_column='predicted_class', scorer_callable=f1_score, metric_name='f1'),
    evaluation_named_tuple(evaluation_column='predicted_class', scorer_callable=balanced_accuracy_score,
                           metric_name='balanced_accuracy_score'),
]
