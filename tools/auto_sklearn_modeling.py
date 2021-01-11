import numpy as np

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import log_loss

from data.db import get_training_data
from helpers.model_helpers import create_x_y_split, create_train_test_split
from modeling.config import TARGET
from modeling.pipelines import get_pipeline

if __name__ == "__main__":
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
    automl = AutoSklearnClassifier(time_left_for_this_task=300, n_jobs=-1, metric=log_loss)
    automl.fit(x_train, y_train)
    print(automl.show_models())
    print()
    predictions = automl.predict_proba(x_test)
    print(log_loss(y_test, predictions[:, 1]))
