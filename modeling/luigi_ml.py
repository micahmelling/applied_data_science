import luigi
import joblib
import numpy as np

from datetime import date
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from tune_sklearn import TuneSearchCV
from sklearn.metrics import log_loss

from data.db import get_training_data
from helpers.model_helpers import create_x_y_split, create_train_test_split
from modeling.config import TARGET, FOREST_PARAM_GRID, ENGINEERING_PARAM_GRID
from modeling.pipelines import get_pipeline


class GetData(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget(
            path='y_test.pkl'
        )

    def run(self):
        print('getting data...')
        df = get_training_data()
        df = df.drop(labels=['client_id', 'id', 'meta__inserted_at'], axis=1)
        df[TARGET] = np.where(df[TARGET] == 'yes', 1, 0)
        x, y = create_x_y_split(df, TARGET)
        x_train, x_test, y_train, y_test = create_train_test_split(x, y)
        joblib.dump(x_train, 'x_train.pkl', compress=3)
        joblib.dump(x_test, 'x_test.pkl', compress=3)
        joblib.dump(y_train, 'y_train.pkl', compress=3)
        joblib.dump(y_test, 'y_test.pkl', compress=3)


class TrainRandomForest(luigi.Task):
    def requires(self):
        return GetData()

    def output(self):
        return luigi.LocalTarget(
            path=(str(date.today()) + '_random_forest.pkl')
        )

    def run(self):
        print('training random forest...')
        x_train = joblib.load('x_train.pkl')
        y_train = joblib.load('y_train.pkl')
        x_test = joblib.load('x_test.pkl')
        y_test = joblib.load('y_test.pkl')
        pipeline = get_pipeline(RandomForestClassifier())
        FOREST_PARAM_GRID.update(ENGINEERING_PARAM_GRID)
        search = TuneSearchCV(pipeline, param_distributions=FOREST_PARAM_GRID, n_trials=50, scoring='neg_log_loss',
                              cv=5, verbose=2, n_jobs=-1, search_optimization='random',
                              early_stopping='MedianStoppingRule')
        search.fit(x_train, y_train)
        best_pipeline = search.best_estimator_
        predictions = best_pipeline.predict_proba(x_test)
        ll_score = log_loss(y_test, predictions[:, 1])
        print(ll_score)
        joblib.dump(best_pipeline, str(date.today()) + '_random_forest.pkl')


class TrainExtraTrees(luigi.Task):
    def requires(self):
        return GetData()

    def output(self):
        return luigi.LocalTarget(
            path=(str(date.today()) + '_extra_trees.pkl')
        )

    def run(self):
        print('training extra trees...')
        x_train = joblib.load('x_train.pkl')
        y_train = joblib.load('y_train.pkl')
        x_test = joblib.load('x_test.pkl')
        y_test = joblib.load('y_test.pkl')
        pipeline = get_pipeline(ExtraTreesClassifier())
        FOREST_PARAM_GRID.update(ENGINEERING_PARAM_GRID)
        search = TuneSearchCV(pipeline, param_distributions=FOREST_PARAM_GRID, n_trials=50, scoring='neg_log_loss',
                              cv=5, verbose=2, n_jobs=-1, search_optimization='random',
                              early_stopping='MedianStoppingRule')
        search.fit(x_train, y_train)
        best_pipeline = search.best_estimator_
        predictions = best_pipeline.predict_proba(x_test)
        ll_score = log_loss(y_test, predictions[:, 1])
        print(ll_score)
        joblib.dump(best_pipeline, str(date.today()) + '_random_forest.pkl')


class TrainModels(luigi.Task):
    def requires(self):
        return TrainExtraTrees(), TrainRandomForest()

    def output(self):
        return luigi.LocalTarget(
            path=(str(date.today()) + '.txt')
        )

    def run(self):
        with open(str(date.today()) + '.txt', 'w') as file:
            file.write('luigi tasks complete')


if __name__ == '__main__':
    luigi.run()
