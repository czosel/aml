import numpy as np
import joblib
from time import time
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.feature_selection import (
    SelectPercentile,
    mutual_info_regression,
    SequentialFeatureSelector,
)
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm

persistent = True
save = False


def prepare_data(X_train_raw, X_test_raw, y_raw, select=True, clip=True):
    def impute(data, reg=None):
        """ Impute missing values. """
        X, X_test, y = data
        if reg is None:
            imp = SimpleImputer(missing_values=np.nan, strategy="mean")
            imp.fit(X)
            return imp.transform(X), imp.transform(X_test), y
        else:
            estimators = {
                "bayes_ridge": BayesianRidge(),
                "dec_tree": DecisionTreeRegressor(max_features="sqrt"),
                "extra_tree": ExtraTreesRegressor(n_estimators=10, random_state=0),
                "KNN": KNeighborsRegressor(n_neighbors=4),
            }
            imp = IterativeImputer(
                estimator=estimators[reg], max_iter=10, tol=0.5, n_nearest_features=100
            )
            tic = time()
            imp.fit(X)
            toc = time()
            print(f"impute done in: {toc - tic:.2f}s")
            return imp.transform(X), imp.transform(X_test), y

    def standardize(data):
        X, X_test, y = data
        scaler = preprocessing.RobustScaler().fit(X)
        return scaler.transform(X), scaler.transform(X_test), y

    n_features = 0.99
    direction = "backward"
    percentile = 15

    def select_percentile(data):
        nonlocal select
        if not select:
            return data

        X, X_test, y = data

        filename = f"joblib/percentile-{n_features}-out-of-{percentile}-percent-direction-{direction}.joblib"
        if persistent and Path(filename).is_file():
            print(f"loading percentile selection from cache: {filename}")
            select = joblib.load(filename)
        else:
            select = SelectPercentile(
                mutual_info_regression, percentile=percentile
            ).fit(X, y)
            if persistent:
                print("saving " + filename)
                joblib.dump(select, filename)

        return select.transform(X), select.transform(X_test), y

    def select_greedy(data):
        X, X_test, y = data

        filename = f"joblib/greedy-{n_features}-out-of-{percentile}-percent-direction-{direction}.joblib"

        if persistent and Path(filename).is_file():
            print(f"loading greedy selection from cache: {filename}")
            select = joblib.load(filename)
        else:
            svr = svm.SVR(kernel="rbf", C=100).fit(X, y)
            tic = time()
            select = SequentialFeatureSelector(
                svr, direction=direction, n_features_to_select=n_features, n_jobs=-1
            ).fit(X, y)
            toc = time()
            print(f"features selected: {select.get_support()}")
            print(f"done in: {toc - tic:.2f}s")

            if persistent and save:
                print("saving " + filename)
                joblib.dump(select, filename)

        return select.transform(X), select.transform(X_test), y

    def clip_outliers(data):
        def _clip_outliers(data, m=3):
            stdev = np.std(data)
            mean = np.mean(data)
            maskMin = mean - stdev * m
            maskMax = mean + stdev * m
            return np.clip(data, maskMin, maskMax)

        if not clip:
            return data

        X, X_test, y = data
        return _clip_outliers(X), _clip_outliers(X_test), y

    def log(data):
        print(len(data))
        print(data)
        return data

    # delete label row and first column
    _X = np.delete(np.delete(X_train_raw, 0, 0), 0, 1)
    _X_test = np.delete(np.delete(X_test_raw, 0, 0), 0, 1)
    y = np.ravel(np.delete(np.delete(y_raw, 0, 0), 0, 1))

    return clip_outliers(select_percentile(standardize(impute((_X, _X_test, y)))))
