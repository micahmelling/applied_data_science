import numpy as np
import time
import os

from ds_helpers import aws

from data.db import get_training_data, make_mysql_connection
from helpers.model_helpers import create_x_y_split, create_train_test_split, create_uid, save_data_in_model_directory
from modeling.config import MODEL_EVALUATION_LIST, CV_SCORER, TARGET, CLASS_CUTOFF, CALIBRATION_PLOT_BINS, CV_FOLDS, \
    TEST_SET_PERCENTAGE, MODEL_TRAINING_LIST, ENGINEERING_PARAM_GRID, DB_SECRET_NAME, DB_SCHEMA_NAME, \
    LOGGING_S3_BUCKET, LOG_TO_DB, DROP_COL_SCORER, DROP_COL_SCORER_STRING, DROP_COL_SCORING_TYPE, \
    DROP_COL_HIGHER_IS_BETTER, EXPLANATION_SAMPLE_N, USE_SHAP_KERNEL
from modeling.model import train_model
from modeling.pipeline import get_pipeline
from modeling.evaluate import run_omnibus_model_evaluation
from modeling.explain import run_omnibus_model_explanation


def create_training_and_testing_data(target, test_set_percentage, db_conn):
    """
    Creates training and testing data for modeling.

    :param target: name of the target
    :param test_set_percentage: percentage of observations for the test set
    :param db_conn: database connection
    :returns: x_train, x_test, y_train, y_test
    """
    print('creating training and testing data...')
    df = get_training_data(db_conn)
    df = df.drop(['client_id', 'id', 'meta__inserted_at'], 1)
    df[TARGET] = np.where(df[target] == 'yes', 1, 0)
    x, y = create_x_y_split(df, target)
    x_train, x_test, y_train, y_test = create_train_test_split(x, y, test_set_percentage)
    return x_train, x_test, y_train, y_test


def train_and_evaluate_model(x_train, x_test, y_train, y_test, model_training_list, cv_strategy, cv_scoring,
                             static_param_space, class_cutoff, target, evaluation_list, calibration_bins,
                             drop_col_scorer, drop_col_scorer_string, drop_col_scoring_type, drop_col_higher_is_better,
                             explanation_sample_n, use_shap_kernel, s3_logging_bucket, db_schema_name=None,
                             db_conn=None, log_to_db=None):
    """
    Trains, evaluates, and explains a series of machine learning models.

    :param x_train: x train
    :param x_test: x test
    :param y_train: y train
    :param y_test: y test
    :param model_training_list: list of named tuples containing model configurations; the following tuple elements are
    required: model_name, model, param_space, iterations
    :param cv_strategy: cross validation strategy
    :param cv_scoring: scoring strategy for cross validation
    :param static_param_space: param space valid for every model
    :param class_cutoff: probability percentage to be classified in the position class
    :param target: name of the target
    :param evaluation_list: list of named tuples containing model evaluation configurations: the following tuple
    elements are required: evaluation_column, scorer_callable, metric_name
    :param calibration_bins: list of calibration bins to show
    :param drop_col_scorer: scikit-learn scoring function for drop col model
    :param drop_col_scorer_string: scoring metric in the form of a string (e.g. 'neg_log-loss') for drop col model
    :param drop_col_scoring_type: either class or probability for drop col model
    :param drop_col_higher_is_better: Boolean of whether or not a higher score is better (e.g. roc auc vs. log loss) for
    drop col model
    :param explanation_sample_n: number of observations to include when performing feature explanation
    :param use_shap_kernel: Boolean of whether or not to use the SHAP kernel explainer
    :param s3_logging_bucket: S3 bucket in which to store the model output
    :param db_schema_name: name of the schema for logging model results
    :param db_conn: database connection
    :param log_to_db: Boolean of whether or not to log results to the database
    """
    for model in model_training_list:
        loop_start_time = time.time()
        model_uid = create_uid(base_string=model.model_name)
        save_data_in_model_directory(model_uid=model_uid, x_train=x_train, x_test=x_test, y_train=y_train,
                                     y_test=y_test)
        best_pipeline = train_model(x_train=x_train, y_train=y_train, get_pipeline_function=get_pipeline,
                                    model_uid=model_uid, model=model.model, param_space=model.param_space,
                                    iterations=model.iterations, cv_strategy=cv_strategy, cv_scoring=cv_scoring,
                                    static_param_space=static_param_space, db_schema_name=db_schema_name,
                                    db_conn=db_conn, log_to_db=log_to_db)
        run_omnibus_model_evaluation(estimator=best_pipeline, model_uid=model_uid, x_test=x_test, y_test=y_test,
                                     class_cutoff=class_cutoff, target=target, evaluation_list=evaluation_list,
                                     calibration_bins=calibration_bins, db_schema_name=db_schema_name, db_conn=db_conn,
                                     log_to_db=log_to_db)
        run_omnibus_model_explanation(estimator=best_pipeline, model_uid=model_uid, x_test=x_test, y_test=y_test,
                                      x_train=x_train,  y_train=y_train, drop_col_scorer=drop_col_scorer,
                                      drop_col_scorer_string=drop_col_scorer_string,
                                      drop_col_scoring_type=drop_col_scoring_type,
                                      drop_col_higher_is_better=drop_col_higher_is_better,
                                      sample_n=explanation_sample_n, use_shap_kernel=use_shap_kernel,
                                      db_schema_name=db_schema_name, db_conn=db_conn, log_to_db=log_to_db)
        print(f'uploading {model_uid} directory to S3...')
        aws.upload_directory_to_s3(local_directory=os.path.join('modeling', model_uid), bucket=s3_logging_bucket)
        print(f'--- {time.time() - loop_start_time} seconds for to train{model_uid} ---')


def main(target, test_set_percentage, model_training_list, cv_strategy, cv_scoring, static_param_space, class_cutoff,
         evaluation_list, calibration_bins, drop_col_scorer, drop_col_scorer_string, drop_col_scoring_type,
         drop_col_higher_is_better, explanation_sample_n, use_shap_kernel, s3_logging_bucket, db_schema_name, log_to_db,
         db_secret_name):
    """
    Main execution function.

    :param target: name of the target
    :param test_set_percentage: percentage of observations for the test set
    :param model_training_list: list of named tuples containing model configurations; the following tuple elements are
    required: model_name, model, param_space, iterations
    :param cv_strategy: cross validation strategy
    :param cv_scoring: scoring strategy for cross validation
    :param static_param_space: param space valid for every model
    :param class_cutoff: probability percentage to be classified in the position class
    :param target: name of the target
    :param evaluation_list: list of named tuples containing model evaluation configurations: the following tuple
    elements are required: evaluation_column, scorer_callable, metric_name
    :param calibration_bins: list of calibration bins to show
    :param drop_col_scorer: scikit-learn scoring function for drop col model
    :param drop_col_scorer_string: scoring metric in the form of a string (e.g. 'neg_log-loss') for drop col model
    :param drop_col_scoring_type: either class or probability for drop col model
    :param drop_col_higher_is_better: Boolean of whether or not a higher score is better (e.g. roc auc vs. log loss) for
    drop col model
    :param explanation_sample_n: number of observations to include when performing feature explanation
    :param use_shap_kernel: Boolean of whether or not to use the SHAP kernel explainer
    :param s3_logging_bucket: S3 bucket in which to store the model output
    :param db_schema_name: name of the schema for logging model results
    :param log_to_db: Boolean of whether or not to log results to the database
    :param db_secret_name: Secrets Manager secret with database credentials
    """
    db_conn = make_mysql_connection(db_secret_name)
    x_train, x_test, y_train, y_test = create_training_and_testing_data(target, test_set_percentage, db_conn)
    train_and_evaluate_model(x_train, x_test, y_train, y_test, model_training_list, cv_strategy, cv_scoring,
                             static_param_space, class_cutoff, target, evaluation_list, calibration_bins,
                             drop_col_scorer, drop_col_scorer_string, drop_col_scoring_type, drop_col_higher_is_better,
                             explanation_sample_n, use_shap_kernel, s3_logging_bucket, db_schema_name, db_conn,
                             log_to_db)


if __name__ == "__main__":
    script_start_time = time.time()
    main(
        target=TARGET,
        test_set_percentage=TEST_SET_PERCENTAGE,
        model_training_list=MODEL_TRAINING_LIST,
        cv_strategy=CV_FOLDS,
        cv_scoring=CV_SCORER,
        static_param_space=ENGINEERING_PARAM_GRID,
        class_cutoff=CLASS_CUTOFF,
        evaluation_list=MODEL_EVALUATION_LIST,
        calibration_bins=CALIBRATION_PLOT_BINS,
        drop_col_scorer= DROP_COL_SCORER,
        drop_col_scorer_string=DROP_COL_SCORER_STRING,
        drop_col_scoring_type=DROP_COL_SCORING_TYPE,
        drop_col_higher_is_better=DROP_COL_HIGHER_IS_BETTER,
        explanation_sample_n=EXPLANATION_SAMPLE_N,
        use_shap_kernel=USE_SHAP_KERNEL,
        s3_logging_bucket=LOGGING_S3_BUCKET,
        db_schema_name=DB_SCHEMA_NAME,
        log_to_db=LOG_TO_DB,
        db_secret_name=DB_SECRET_NAME
    )
    print("--- {} seconds for script to run ---".format(time.time() - script_start_time))
