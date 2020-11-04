from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import sklearn
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import regularizers

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

batch_size = 10


class WideAndDeepNetClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, mode="ava", nfirst=2, activation="relu", regularization="dropout", weights=[6.0,1.0,6.0]):
        """
        Called when initializing the classifier
        """
        # if weights is None:
        #     weights = [6.0, 1.0, 6.0]
        self.mode = mode
        self.nfirst = nfirst
        self.activation = activation
        self.regularization = regularization
        self.weights = weights

        def weighted_loss(weights):
            def weighted_categ_crossentropy(onehot_labels, logits, weights=weights):
                class_weights = tf.compat.v2.constant([weights])
                weights = tf.compat.v2.reduce_sum(class_weights * onehot_labels, axis=1)
                unweighted_losses = tf.compat.v2.nn.softmax_cross_entropy_with_logits(onehot_labels, logits)
                return tf.reduce_mean(unweighted_losses * weights)

            return weighted_categ_crossentropy

        def _model():
            lin_mod = tf.keras.experimental.LinearModel(3)
            dnn_mod = tf.keras.Sequential()
            dnn_mod.add(tf.keras.layers.Dense(128, activation=activation))
            dnn_mod.add(tf.keras.layers.Dense(64, activation=activation))
            dnn_mod.add(tf.keras.layers.Dense(3, activation='sigmoid'))
            # dnn_mod = tf.keras.experimental.LinearModel(3, activation=activation)
            model = tf.keras.experimental.WideDeepModel(lin_mod, dnn_mod)

            # "KERNEL ANN"
            # model = tf.keras.Sequential([
            #     tf.keras.Input(shape=1000,),
            #     tf.keras.layers.experimental.RandomFourierFeatures(
            #         output_dim=4096,
            #         scale=10.,
            #         trainable=True,
            #         kernel_initializer='laplacian'),
            #     tf.keras.layers.Dense(units=3, activation='sigmoid'),
            # ])
            # model = tf.keras.experimental.WideDeepModel(lin_mod, model)

            # model.add(tf.keras.layers.Dense(3, activation='sigmoid'))
            model.compile(loss=weighted_loss(weights))
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
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)


        # history = self.model_fn().fit(X_train, to_categorical(y_train),
        #                               validation_data=(X_test, to_categorical(y_test)), callbacks=[callback],
        #                               shuffle=True, epochs=500, batch_size=32, verbose=0)
        # nepochs = len(history.history) - patience
        #
        # self.model.fit(X, to_categorical(y), epochs=nepochs, verbose=0)
        # # self.model.summary()
        # return self
        self.model.fit(X_train, to_categorical(y_train), validation_data=(X_test, to_categorical(y_test)),
                                      callbacks=[callback], shuffle=True, epochs=500, batch_size=32, verbose=0)
        # self.model.summary()
        return self


    def predict(self, X, y=None):
        return np.argmax(self.model.predict(X), axis=1)
        # return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        print(y.shape)
        print(np.argmax(self.predict(X), axis=1).shape)
        print(self.predict(X).shape)
        return sklearn.metrics.balanced_accuracy_score(y, np.argmax(self.predict(X), axis=1))
