from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import sklearn
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import regularizers

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

batch_size = 10


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, ncells=3, activation="relu"):
        """
        Called when initializing the classifier
        """
        # if weights is None:
        #     weights = [6.0, 1.0, 6.0]
        self.ncells = ncells
        self.activation = activation


        def _model():
            model = tf.keras.Sequential()
            # model.add(tf.keras.layers.LSTM(3, activation=activation, input_shape=(159,1)))
            model.add(tf.keras.layers.LSTM(ncells, activation=activation, batch_input_shape=(1, None, 1)))
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            model.compile(loss="binary_crossentropy")
            return model

        self.model_fn = _model
        self.model = self.model_fn()

    def model(self, intValu):
        return None

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

        patience = 10
        #estimate no. epochs to train by single sample
        # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # history = self.model_fn().fit(X_train, to_categorical(y_train),
        #                               validation_data=(X_test, to_categorical(y_test)), callbacks=[callback],
        #                               shuffle=True, epochs=500, batch_size=32, verbose=0)
        # nepochs = len(history.history) - patience
        #
        # self.model.fit(X, to_categorical(y), epochs=nepochs, verbose=0)
        # # self.model.summary()
        # return self
        lastscore = 10
        score = 9.999
        self.thresh = 0
        patient = 0
        raw_pred = None
        while score < lastscore or patient < 1:
            order = np.arange(len(X_train))
            np.random.shuffle(order)
            if score >= lastscore:
                patient += 1
            for j in order:
                self.model.fit(np.array(X_train[j][~np.isnan(X_train[j])]).reshape((1,-1,1)), np.array(y_train[j]).reshape((1,1)), batch_size=1, verbose=0)
            raw_pred = np.array([self.model.predict(np.array(X_test[i][~np.isnan(X_test[i])]).reshape((1,-1,1)) for i in range(len(X_test)))]).flatten()
            loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(y_test, raw_pred).numpy()

            print("val loss:", loss)
            lastscore = score
            if loss < score:
                patient = 0
                score = loss

        # lastscore = score
        score = 0
        for t in np.arange(0,0.7,0.025):
            y_pred = [1 if x > t else 0 for x in raw_pred]
            _score = sklearn.metrics.f1_score(y_test, y_pred)
            if _score > score:
                score = _score
                self.thresh = t
        print("score:", score, " thresh", self.thresh)
        # self.model.fit([[X_train[i]]], [y_train[i]], validation_data=(X_test, to_categorical(y_test)),
        #                callbacks=[callback], shuffle=True, epochs=500, batch_size=1, verbose=2)
        # self.model.summary()
        return self


    def predict(self, X, y=None):
        raw_pred = np.array([self.model.predict(
            np.array(X[i][~np.isnan(X[i])]).reshape((1, -1, 1)) for i in range(len(X)))]).flatten()

        return [1 if x > self.thresh else 0 for x in raw_pred]

    def score(self, X, y, sample_weight=None):
        print(y.shape)
        print(np.argmax(self.predict(X), axis=1).shape)
        print(self.predict(X).shape)
        return sklearn.metrics.balanced_accuracy_score(y, np.argmax(self.predict(X), axis=1))
