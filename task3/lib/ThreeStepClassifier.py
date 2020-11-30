from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import sklearn
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import regularizers

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

batch_size = 10


class ThreeStepClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, classifiers=[]):
        """
        Called when initializing the classifier
        """
        # if weights is None:
        #     weights = [6.0, 1.0, 6.0]
        self.classifiers = classifiers
        self.classes_ = [0,1,2,3]



    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        y3 = y.copy()
        y3[y != 3] = 0
        y3[y == 3] = 1
        self.classifiers[0].fit(X, y3)


        y1 = y.copy()
        y1[y == 2] = 0
        self.classifiers[1].fit(X[y != 3], y1[y != 3])

        i0 = (y == 0)+(y == 2)
        y0 = y.copy()
        y0[y == 2] = 1
        self.classifiers[2].fit(X[i0], y0[i0])
        return self


    def predict(self, X, y=None):
        y3 = self.classifiers[0].predict(X)
        y1 = self.classifiers[1].predict(X)
        y0 = self.classifiers[2].predict(X)
        return [3 if y3[i] == 1 else 1 if y1[i] == 1 else 2 if y0[i] == 1 else 0 for i in range(len(X))]
        # return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        print(y.shape)
        print(np.argmax(self.predict(X), axis=1).shape)
        print(self.predict(X).shape)
        return sklearn.metrics.balanced_accuracy_score(y, np.argmax(self.predict(X), axis=1))
