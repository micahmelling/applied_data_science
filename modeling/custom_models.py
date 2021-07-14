import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array


class HeuristicClassifier(BaseEstimator, ClassifierMixin):
    """
    Models the outcome as a function of the row sum. The higher the row sum, the higher the probability of being in
    the positive class. The probabilities are throttled by the max parameter. Increasing this parameter will lower
    the probabilities and make the model more conservative and fewer positive cases would be predicted.
    """
    def __init__(self, maximum=250):
        self.maximum = maximum
        self.min = 0

    def fit(self, X, y=None):
        """
        No fit is required for this model. We simply check data here.
        """
        assert (type(self.maximum) == int), 'max parameter must be integer'
        X = check_array(X)
        X, y = check_X_y(X, y)
        return self

    @staticmethod
    def _calculate_row_sum(X):
        """
        Calculates the sum of the row.
        """
        return X.sum(axis=1)

    @staticmethod
    def _create_proba(sum_array, max):
        """
        Maps an array into a probability, based on the maximum parameter.
        """
        proba = sum_array / max
        proba = np.clip(proba, 0.0001, 0.9999)
        return proba

    def predict(self, X, y=None):
        """
        Converts probability estimates into a 0-1 classification, based on a 0.50 threshold.
        """
        X = check_array(X)
        sum_array = self._calculate_row_sum(X)
        proba = self._create_proba(sum_array, self.maximum)
        return np.digitize(proba, np.array([0.50, 1.0]))

    def predict_proba(self, X, y=None):
        """
        Returns predicted probabilities.
        """
        X = check_array(X)
        sum_array = self._calculate_row_sum(X)
        proba = self._create_proba(sum_array, self.maximum)
        return np.array([1 - proba, proba]).T

    def score(self, X, y, sample_weight=None):
        """
        Supplies the accuracy of the model
        """
        return accuracy_score(y, self.predict(X))
