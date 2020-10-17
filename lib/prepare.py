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
    SelectFromModel,
    SequentialFeatureSelector,
)
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn import svm

# import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV


def prepare_data(X_train_raw, X_test_raw, y_raw, prepare_config):
    n_features = prepare_config.get("n_features", 1)
    direction = prepare_config.get("direction", "forward")
    percentile = prepare_config.get("percentile", 20)
    n_neighbors = prepare_config.get("n_neighbors", 40)
    outlier_algo = prepare_config.get("outlier_algo", "auto")
    contamination = prepare_config.get("contamination", "auto")
    n_estimators = prepare_config.get("n_estimators", 100)
    m = prepare_config.get("m", 3)

    def impute(data, reg=None):
        """ Impute missing values. """
        X, X_test, y = data
        if reg is None:
            imp = SimpleImputer(missing_values=np.nan, strategy="median")
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

    def select_percentile(data):
        X, X_test, y = data

        select = SelectPercentile(mutual_info_regression, percentile=percentile).fit(
            X, y
        )

        return select.transform(X), select.transform(X_test), y

    def select_model(data):
        X, X_test, y = data

        lasso = LassoCV(tol=1).fit(X, y)
        importance = np.abs(lasso.coef_)
        # plt.bar(height=importance, x=range(0, len(importance)))
        # plt.title("Feature importances via coefficients")
        # plt.show()

        sfm = SelectFromModel(lasso, threshold=0.01).fit(X, y)
        print(
            "Features selected by SelectFromModel: "
            f"{np.where(sfm.get_support() == True)[0].size}"
        )

        return sfm.transform(X), sfm.transform(X_test), y

    def select_greedy(data):
        X, X_test, y = data

        svr = svm.SVR(kernel="rbf", C=100, tol=1).fit(X, y)
        tic = time()
        select = SequentialFeatureSelector(
            svr, direction=direction, n_features_to_select=n_features, n_jobs=-1
        ).fit(X, y)
        toc = time()
        print(f"features selected: {select.get_support()}")
        print(f"done in: {toc - tic:.2f}s")

        return select.transform(X), select.transform(X_test), y

    def dump_outliers(data):
        algos = {
            "isolation_forest": IsolationForest(
                n_estimators=n_estimators, contamination=contamination
            ),
            "local_outlier_factor": LocalOutlierFactor(
                n_neighbors, contamination=contamination
            ),
            "svm": svm.OneClassSVM(nu=contamination, kernel="rbf"),
        }
        X, X_test, y = data
        clf = algos[outlier_algo]
        if outlier_algo == "local_outlier_factor":
            y_pred = clf.fit_predict(X)
        else:
            y_pred = clf.fit(X).predict(X)

        # print(f"removing {np.where(y_pred < 0)[0].size} outliers")
        return X[y_pred > 0], X_test, y[y_pred > 0]

    def clip_outliers(data):
        def _clip_outliers(data):
            stdev = np.std(data)
            mean = np.mean(data)
            maskMin = mean - stdev * m
            maskMax = mean + stdev * m
            return np.clip(data, maskMin, maskMax)

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

    return clip_outliers(
        dump_outliers(
            select_greedy(select_model(standardize(impute((_X, _X_test, y)))))
        )
    )
