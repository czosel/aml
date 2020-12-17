import numpy as np
import itertools


def load_csv(path, limit=None):
    if limit:
        with open(path) as f:
            return np.genfromtxt(itertools.islice(f, limit), dtype=float, delimiter=",")

    return np.genfromtxt(path, dtype=float, delimiter=",")


def clean(data):
    """Delete label row and first column."""

    X, X_test, y = data
    X = tuple(np.delete(np.delete(x, 0, 0), 0, 1) for x in X)
    X_test = tuple(np.delete(np.delete(x, 0, 0), 0, 1) for x in X_test)
    y = np.ravel(np.delete(np.delete(y, 0, 0), 0, 1))
    print("training data", X[0].shape)
    print("test data", X_test[0].shape)
    return X, X_test, y


def load_data(limit=None):
    X_eeg1 = load_csv("data/train_eeg1.csv", limit)
    X_eeg2 = load_csv("data/train_eeg2.csv", limit)
    X_emg = load_csv("data/train_emg.csv", limit)
    X_test_eeg1 = load_csv("data/test_eeg1.csv", limit)
    X_test_eeg2 = load_csv("data/test_eeg2.csv", limit)
    X_test_emg = load_csv("data/test_emg.csv", limit)
    y = load_csv("data/train_labels.csv", limit)

    return clean(((X_eeg1, X_eeg2, X_emg), (X_test_eeg1, X_test_eeg2, X_test_emg), y))
