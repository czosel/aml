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


def select(data):
    # fmt: off
    good_features = [  7,  30,  35,  68,  76,  81,  85, 101, 121, 123, 127, 134, 169,
           175, 177, 184, 196, 198, 204, 210, 227, 239, 240, 241, 247, 262,
           275, 285, 287, 289, 296, 302, 309, 343, 353, 359, 360, 375, 400,
           403, 408, 411, 416, 417, 423, 432, 434, 447, 448, 457, 464, 476,
           477, 494, 496, 498, 504, 508, 523, 537, 559, 566, 578, 579, 581,
           597, 599, 661, 662, 669, 685, 715, 717, 753, 772, 776, 785, 800,
           819, 823]
    # fmt: on
    X, X_test, y = data
    return np.take(X, good_features, 1), np.take(X_test, good_features, 1), y


def load_data():
    X = load_csv("data/X_train.csv")
    X_test = load_csv("data/X_test.csv")
    y = load_csv("data/y_train.csv")

    return select(clean((X, X_test, y)))
