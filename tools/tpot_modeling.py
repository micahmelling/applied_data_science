import numpy as np

from tpot import TPOTClassifier

from data.db import get_training_data
from helpers.model_helpers import create_x_y_split, create_train_test_split
from modeling.config import TARGET, CV_SCORER, CV_TIMES
from modeling.pipelines import get_pipeline


def run_tpot():
    df = get_training_data()
    df = df.drop(labels=['client_id', 'id', 'meta__inserted_at'], axis=1)
    df[TARGET] = np.where(df[TARGET] == 'yes', 1, 0)
    x, y = create_x_y_split(df, TARGET)
    x_train, x_test, y_train, y_test = create_train_test_split(x, y)
    pipeline = get_pipeline(model=None)
    pipeline.steps.pop(len(pipeline) - 1)
    pipeline.steps.pop(len(pipeline) - 1)
    pipeline.steps.pop(len(pipeline) - 1)
    x_train = pipeline.fit_transform(x_train)
    x_test = pipeline.transform(x_test)
    tpot = TPOTClassifier(generations=5, population_size=50, scoring=CV_SCORER, cv=CV_TIMES, early_stop=3,
                          max_time_mins=10, verbosity=3, random_state=42, n_jobs=-1)
    tpot.fit(x_train, y_train)
    print(tpot.score(x_test, y_test))
    tpot.export('tpot_pipeline.py')


if __name__ == "__main__":
    run_tpot()
