import joblib
import os
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV

from modeling.config import MODELS_DIRECTORY, DIAGNOSTICS_DIRECTORY


# TODO: tunesearchcv ... regular hyperopt
def train_model(x_train, y_train, get_pipeline_function, model_name, model, param_space, n_trials, scoring):
    """

    """
    print(f'training {model_name}...')
    pipeline = get_pipeline_function(model)
    search = RandomizedSearchCV(pipeline, param_distributions=param_space, n_iter=n_trials, scoring=scoring, n_jobs=-1)
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    cv_results = pd.DataFrame(search.cv_results_).sort_values(by=['rank_test_score'], ascending=False)
    joblib.dump(best_model, os.path.join(MODELS_DIRECTORY, f'{model_name}.pkl'))
    cv_results.to_csv(os.path.join(DIAGNOSTICS_DIRECTORY, f'{model_name}_cv_results.csv'), index=False)
    return best_model
