import os
import joblib
import numpy as np
import time

from sklearn.metrics import log_loss

from data.db import get_training_data
from helpers.model_helpers import create_x_y_split, create_train_test_split, create_model_uid, \
    make_directories_if_not_exists, upload_model_directory_to_s3
from modeling.config import MODEL_TRAINING_DICT, MODEL_EVALUATION_LIST, CV_SCORER, TARGET, MODELS_DIRECTORY, \
    DIAGNOSTICS_DIRECTORY, DATA_DIRECTORY, CLASS_CUTOFF, CALIBRATION_BINS, CV_TIMES, TEST_SET_PERCENTAGE
from modeling.model import train_model
from modeling.pipeline import get_pipeline
# from modeling.evaluate import run_omnibus_model_evaluation


def train():
    """
    Trains and evaluates machine learning models.
    """
    script_start_time = time.time()
    # df = get_training_data()
    # df = df.drop(labels=['client_id', 'id', 'meta__inserted_at'], axis=1)
    # df[TARGET] = np.where(df[TARGET] == 'yes', 1, 0)
    # x, y = create_x_y_split(df, TARGET)
    # x_train, x_test, y_train, y_test = create_train_test_split(x, y, test_size=TEST_SET_PERCENTAGE)
    x_train = joblib.load('extra_trees_202107122057549185780500/data/x_train.pkl')
    x_train = x_train.head(5_000)
    x_test = joblib.load('extra_trees_202107122057549185780500/data/x_test.pkl')
    x_test = x_test.head(5_000)
    y_train = joblib.load('extra_trees_202107122057549185780500/data/y_train.pkl')
    y_train = y_train.head(5_000)
    y_test = joblib.load('extra_trees_202107122057549185780500/data/y_test.pkl')
    y_test = y_test.head(5_000)
    for model_name, model_config in MODEL_TRAINING_DICT.items():
        loop_start_time = time.time()
        model_name = create_model_uid(model_name)
        model_directories = [os.path.join(model_name, MODELS_DIRECTORY),
                             os.path.join(model_name, DIAGNOSTICS_DIRECTORY),
                             os.path.join(model_name, DATA_DIRECTORY)]
        make_directories_if_not_exists(model_directories)
        joblib.dump(x_train, os.path.join(model_name, DATA_DIRECTORY, 'x_train.pkl'), compress=9)
        joblib.dump(x_test, os.path.join(model_name, DATA_DIRECTORY, 'x_test.pkl'), compress=9)
        joblib.dump(y_train, os.path.join(model_name, DATA_DIRECTORY, 'y_train.pkl'), compress=9)
        joblib.dump(y_test, os.path.join(model_name, DATA_DIRECTORY, 'y_test.pkl'), compress=9)
        pipeline = train_model(x_train, y_train, get_pipeline, model_name, model_config[0], model_config[1],
                               model_config[2], CV_TIMES, CV_SCORER)
        # run_omnibus_model_evaluation(pipeline, model_name, x_test, y_test, CLASS_CUTOFF, TARGET, MODEL_EVALUATION_LIST,
        #                              CALIBRATION_BINS)
        # upload_model_directory_to_s3(model_name)

        print("--- {0} seconds for {1} loop to run ---".format(time.time() - loop_start_time, model_name))
    print("--- {} seconds for script to run ---" .format(time.time() - script_start_time))


if __name__ == "__main__":
    train()

    # TODO: need to add average_stars clipper ... need to see if the clipper actually works
    # TODO: parallelize upload_model_directory_to_s3() with mp and glob ... make name more generic
    # TODO: training data becomes sparse matrix for upload

    # from modeling.explain import run_omnibus_model_explanation
    # pipeline = joblib.load('extra_trees_sigmoid_202101172351171374000600/models/model.pkl')
    # x_test = joblib.load('extra_trees_sigmoid_202101172351171374000600/data/x_test.pkl')
    # x_train = joblib.load('extra_trees_sigmoid_202101172351171374000600/data/x_train.pkl')
    # y_test = joblib.load('extra_trees_sigmoid_202101172351171374000600/data/y_test.pkl')
    # y_train = joblib.load('extra_trees_sigmoid_202101172351171374000600/data/y_train.pkl')
    #
    # run_omnibus_model_explanation(
    #     pipeline,
    #     x_test,
    #     y_test,
    #     x_train,
    #     y_train,
    #     scorer=log_loss,
    #     scorer_string='neg_log_loss',
    #     scoring_type='probability',
    #     model_uid='extra_trees_sigmoid_202101172351171374000600',
    #     higher_is_better=True,
    #     calibrated_model=True,
    #     sample_n=100,
    #     use_kernel=True
    # )
