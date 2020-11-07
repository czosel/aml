from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import sklearn
from sklearn.svm import LinearSVC, SVC
import numpy as np
from tensorflow.keras import regularizers

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

batch_size = 10


class OVOClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, c1=None, c2=None, c3=None):
        """
        Called when initializing the classifier
        """
        # if weights is None:
        #     weights = [6.0, 1.0, 6.0]
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        predictors = [self.c1, self.c2, self.c3]
        for i in range(3):
            indices = np.where(y!=i)
            y_new = y.copy()[indices]
            X_new = X.copy()[indices]
            if i == 0:
                y_new[y_new == 2] = 0
            if i == 1:
                y_new[y_new == 2] = 1

            predictors[i].fit(X_new, y_new)


        return self


    def predict(self, X, y=None):
        p1 = self.c1.predict(X)
        p2 = self.c2.predict(X)
        p3 = self.c3.predict(X)


        # for i in range(5):
        #     print(p1[i], p2[i], p3[i])

        p1[p1 == 0] = 2
        p2[p2 == 1] = 2

        # for i in range(5):
        #     print(".....")
        #     print(p1[i], p2[i], p3[i])

        def to_discreet(l1,l2,l3):
            return 0 if l2 == 0 and l3 == 0 else (1 if l1 == 1 and l3 == 1 else 2)

        return [to_discreet(p1[i], p2[i], p3[i]) for i in range(len(p1))]

        # return y

    def score(self, X, y, sample_weight=None):
        raise Exception()
        print(y.shape)
        print(np.argmax(self.predict(X), axis=1).shape)
        print(self.predict(X).shape)
        return sklearn.metrics.balanced_accuracy_score(y, np.argmax(self.predict(X), axis=1))
