import os


AWS_KEYS_SECRET = 'churn-api-s3-keys'
DATABASE_SECRET = 'churn-model-mysql'
EMAIL_SECRET = 'churn-email'
APP_SECRET = os.environ['CHURN_APP_SECRET']
S3_BUCKET_NAME = 'churn-model-data-science-logs'
SCHEMA_NAME = 'churn_model'
STAGE_URL = 'stage_url'
PROD_URL = 'prod_url'
MODEL_1_PATH = os.path.join('modeling', 'extra_trees_sigmoid_202101172351171374000600', 'models', 'model.pkl')
MODEL_2_PATH = os.path.join('modeling', 'extra_trees_sigmoid_202101172351171374000600', 'models', 'model.pkl')
HEURISTIC_MODEL_PATH = os.path.join('modeling', 'heuristic_model_202101272206199889570600', 'models', 'model.pkl')
MODEL_FEATURES = ['activity_score', 'propensity_score', 'profile_score_new', 'completeness_score', 'xp_points',
                  'profile_score', 'portfolio_score', 'mouse_movement', 'average_stars', 'ad_target_group',
                  'marketing_message', 'device_type', 'all_star_group', 'mouse_x', 'coupon_code', 'ad_engagement_group',
                  'user_group', 'browser_type', 'email_code', 'marketing_creative', 'secondary_user_group',
                  'promotion_category', 'marketing_campaign', 'mouse_y', 'marketing_channel', 'marketing_creative_sub',
                  'site_level', 'acquired_date']
