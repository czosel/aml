from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import sklearn
from sklearn.svm import LinearSVC, SVC
import numpy as np
from tensorflow.keras import regularizers

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

batch_size = 10


class MixedClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, c1=None, c2=None, c3=None, c_tie=None):
        """
        Called when initializing the classifier
        """
        # if weights is None:
        #     weights = [6.0, 1.0, 6.0]
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c_tie = c_tie

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        # predictors = [self.c1, self.c2, self.c3]
        # for i in range(3):
        #     y_new = y.copy()
        #     y_new[y_new == i] = -1
        #     y_new[y_new != -1] = 1
        #     y_new[y_new == -1] = 0
        #
        #     predictors[i].fit(X, y_new)
        #

        self.c_tie.fit(X,y)
        return self


    def predict(self, X, y=None):
        p1 = self.c1.predict(X)
        p2 = self.c2.predict(X)
        p3 = self.c3.predict(X)
        tie = self.c_tie.predict(X)

        def agree(l1,l2,l3):
            return l1+l2+l3 == 1

        def to_discreet(l1,l2,l3):
            return l2*1.0+l3*2.0

        # return [to_discreet(p1[i], p2[i], p3[i]) if agree(p1[i], p2[i], p3[i]) else tie[i] for i in range(len(p1))]

        return y

    def score(self, X, y, sample_weight=None):
        print(y.shape)
        print(np.argmax(self.predict(X), axis=1).shape)
        print(self.predict(X).shape)
        return sklearn.metrics.balanced_accuracy_score(y, np.argmax(self.predict(X), axis=1))
