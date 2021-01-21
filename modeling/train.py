import os
import joblib
import numpy as np
import time

from data.db import get_training_data
from helpers.model_helpers import create_x_y_split, create_train_test_split, create_model_uid, \
    make_directories_if_not_exists, upload_model_directory_to_s3
from modeling.config import MODEL_TRAINING_DICT, MODEL_EVALUATION_LIST, CV_SCORER, TARGET, MODELS_DIRECTORY, \
    DIAGNOSTICS_DIRECTORY, DATA_DIRECTORY, CLASS_CUTOFF, CALIBRATION_BINS, CV_TIMES
from modeling.model import train_model, calibrate_fitted_model
from modeling.pipelines import get_pipeline
from modeling.evaluate import run_omnibus_model_evaluation


def train():
    """
    Trains and evaluates machine learning models.
    """
    script_start_time = time.time()
    df = get_training_data()
    df = df.drop(labels=['client_id', 'id', 'meta__inserted_at'], axis=1)
    df[TARGET] = np.where(df[TARGET] == 'yes', 1, 0)
    x, y = create_x_y_split(df, TARGET)
    x_train, x_test, y_train, y_test = create_train_test_split(x, y, test_size=0.30)
    x_test, x_validation, y_test, y_validation = create_train_test_split(x_test, y_test, test_size=0.15)
    for model_name, model_config in MODEL_TRAINING_DICT.items():
        loop_start_time = time.time()
        model_name = create_model_uid(model_name)
        model_directories = [os.path.join(model_name, MODELS_DIRECTORY),
                             os.path.join(model_name, DIAGNOSTICS_DIRECTORY),
                             os.path.join(model_name, DATA_DIRECTORY)]
        make_directories_if_not_exists(model_directories)
        joblib.dump(x_train, os.path.join(model_name, DATA_DIRECTORY, 'x_train.pkl'), compress=3)
        joblib.dump(x_test, os.path.join(model_name, DATA_DIRECTORY, 'x_test.pkl'), compress=3)
        joblib.dump(y_train, os.path.join(model_name, DATA_DIRECTORY, 'y_train.pkl'), compress=3)
        joblib.dump(y_test, os.path.join(model_name, DATA_DIRECTORY, 'y_test.pkl'), compress=3)
        joblib.dump(x_validation, os.path.join(model_name, DATA_DIRECTORY, 'x_validation.pkl'), compress=3)
        joblib.dump(y_validation, os.path.join(model_name, DATA_DIRECTORY, 'y_validation.pkl'), compress=3)
        pipeline = train_model(x_train, y_train, get_pipeline, model_name, model_config[0], model_config[1],
                               model_config[2], CV_TIMES, CV_SCORER)
        run_omnibus_model_evaluation(pipeline, model_name, x_test, y_test, CLASS_CUTOFF, TARGET, MODEL_EVALUATION_LIST,
                                     CALIBRATION_BINS, False)
        # upload_model_directory_to_s3(model_name)

        try:
            if 'extra_trees_uncalibrated' in model_name:
                model_name = 'extra_trees_post_hoc_sigmoid'
                model_directories = [os.path.join(model_name, MODELS_DIRECTORY),
                                     os.path.join(model_name, DIAGNOSTICS_DIRECTORY),
                                     os.path.join(model_name, DATA_DIRECTORY)]
                make_directories_if_not_exists(model_directories)
                calibrated_pipeline = calibrate_fitted_model(pipeline, x_validation, y_validation)
                run_omnibus_model_evaluation(calibrated_pipeline, model_name, x_test, y_test, CLASS_CUTOFF, TARGET,
                                             MODEL_EVALUATION_LIST,
                                             CALIBRATION_BINS, True)

                model_name = 'extra_trees_post_hoc_isotonic'
                model_directories = [os.path.join(model_name, MODELS_DIRECTORY),
                                     os.path.join(model_name, DIAGNOSTICS_DIRECTORY),
                                     os.path.join(model_name, DATA_DIRECTORY)]
                make_directories_if_not_exists(model_directories)
                calibrated_pipeline = calibrate_fitted_model(pipeline, x_validation, y_validation)
                run_omnibus_model_evaluation(calibrated_pipeline, model_name, x_test, y_test, CLASS_CUTOFF, TARGET,
                                             MODEL_EVALUATION_LIST,
                                             CALIBRATION_BINS, True)
        except:
            pass

        print("--- {0} seconds for {1} loop to run ---".format(time.time() - loop_start_time, model_name))
    print("--- {} seconds for script to run ---" .format(time.time() - script_start_time))


if __name__ == "__main__":
    # train()

    import pandas as pd
    from modeling.evaluate import calculate_probability_lift

    predictions_df = pd.read_csv('extra_trees_sigmoid_202101172351171374000600/diagnostics/extra_trees_sigmoid_202101172351171374000600_predictions.csv')
    pipeline = joblib.load('extra_trees_sigmoid_202101172351171374000600/models/extra_trees_sigmoid_202101172351171374000600.pkl')

    calculate_probability_lift(predictions_df['churn'], predictions_df['1_prob'],
                               'extra_trees_sigmoid_202101172351171374000600')
