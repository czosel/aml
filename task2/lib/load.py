import numpy as np


def load_csv(path):
    return np.genfromtxt(path, dtype=float, delimiter=",")


def clean(data):
    """Delete label row and first column."""

    X, X_test, y = data
    X = np.delete(np.delete(X, 0, 0), 0, 1)
    X_test = np.delete(np.delete(X_test, 0, 0), 0, 1)
    y = np.ravel(np.delete(np.delete(y, 0, 0), 0, 1))
    print("training data", X.shape)
    print("test data", X_test.shape)
    return X, X_test, y


def load_data():
    X = load_csv("data/X_train.csv")
    X_test = load_csv("data/X_test.csv")
    y = load_csv("data/y_train.csv")

    return clean((X, X_test, y))
