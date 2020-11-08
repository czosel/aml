from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import sklearn
from sklearn.svm import LinearSVC, SVC
import numpy as np
from tensorflow.keras import regularizers
from sklearn.dummy import DummyClassifier
from sklearn.mixture import GaussianMixture
import collections

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

batch_size = 10


class HardClusterClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, all_X = None, n_clusters=2, classifiers=None):
        """
        Called when initializing the classifier
        """
        self.classifiers = classifiers
        self.n_clusters = n_clusters
        self.all_X = all_X

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        assert(len(self.classifiers) == self.n_clusters)

        self.mx = GaussianMixture(n_components=self.n_clusters)
        self.mx.fit(self.all_X)
        pred = self.mx.predict(X)
        order = sorted(range(self.n_clusters), key=lambda x: len(np.where(pred == x)[0]), reverse=True)
        print(order)
        self.classifiers = [self.classifiers[i] for i in order]
        for i in range(len(self.classifiers)):
            print("Cluster: ", i)
            indices = pred == i
            y_new = y[indices]
            cnts = collections.Counter(y_new)
            print(cnts)
            if len(y_new) == 0:
                self.classifiers[i] = DummyClassifier(strategy="constant", constant=1).fit(X,y)
            else:
                print("Majority ratio:", cnts[1] / sum(cnts.values()))
                if cnts[1] / sum(cnts.values()) > 0.99:
                    self.classifiers[i] = DummyClassifier(strategy="constant", constant=1).fit(X,y)
                elif len(set(y_new)) == 1:
                    self.classifiers[i] = DummyClassifier(strategy="constant", constant=y_new[0]).fit(X,y)
                else:
                    self.classifiers[i].fit(X[indices,:], y_new)

        return self


    def predict(self, X, y=None):
        clusters = self.mx.predict(X)
        pred = np.array([c.predict(X) for c in self.classifiers])
        return [pred[clusters[i], i] for i in range(len(clusters))]

    def score(self, X, y, sample_weight=None):
        raise Exception()
        print(y.shape)
        print(np.argmax(self.predict(X), axis=1).shape)
        print(self.predict(X).shape)
        return sklearn.metrics.balanced_accuracy_score(y, np.argmax(self.predict(X), axis=1))
