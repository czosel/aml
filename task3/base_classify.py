import numpy as np
import pandas as pd
import json
from tabulate import tabulate
from slugify import slugify
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    FunctionTransformer,
    QuantileTransformer,
    PowerTransformer,
)
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
from sklearn.feature_selection import (
    SelectFromModel,
    SelectPercentile,
    mutual_info_regression,
)
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklego.mixture import GMMOutlierDetector, BayesianGMMOutlierDetector
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor, XGBClassifier
from lib.load import load_data
from lib.ThreeStepClassifier import ThreeStepClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.utils.fixes import loguniform
from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA, KernelPCA

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# def load_data():

X, X_test, y = np.loadtxt("features/cache/train_X.csv", delimiter=","), np.loadtxt("features/cache/test_X.csv", delimiter=","), np.loadtxt("features/cache/train_y.csv", delimiter=",")
X_extra1, X_test_extra1 = np.loadtxt("features/LSTMCache/y_feat.csv", delimiter=","), np.loadtxt("features/LSTMCache/y_test.csv", delimiter=",")
X_extra2, X_test_extra2 = np.loadtxt("features/LGBMCache/y_feat.csv", delimiter=","), np.loadtxt("features/LGBMCache/y_test.csv", delimiter=",")

print(X.shape, X_test.shape)
print(X_extra1.shape, X_test_extra1.shape)
print(X_extra2.shape, X_test_extra2.shape)

X = np.concatenate((X,X_extra1, X_extra2), axis=1)
X_test = np.concatenate((X_test,X_test_extra1, X_test_extra2), axis=1)

print(X.shape, X_test.shape)

fitting = Pipeline(
    [
        ("impute", sklearn.impute.SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("LGBM", LGBMClassifier(n_jobs=-1, num_class=4))
        # ("LGBM", ThreeStepClassifier([LGBMClassifier(n_jobs=-1),LGBMClassifier(n_jobs=-1),LGBMClassifier(n_jobs=-1)]))
    ],
    memory="cache",
)

param_distributions = {
    "LGBM__num_leaves": [100],
    "LGBM__min_data_in_leaf": [30],
    "LGBM__objective": ["multiclassova"]

}
search = RandomizedSearchCV(
    fitting,
    param_distributions,
    scoring="f1_micro",
    n_iter=20,
    n_jobs=1,
    verbose=2,
).fit(X, y)
print(f"F1_MICRO score={search.best_score_:.3f}): {search.best_params_}")
frame = pd.DataFrame.from_dict(search.cv_results_)
print(
    tabulate(
        frame[["params", "mean_test_score", "std_test_score", "rank_test_score"]],
        headers="keys",
    )
)

print("classification_report of best estimator:")
print(classification_report(y, search.best_estimator_.predict(X)))
print(pd.DataFrame(search.cv_results_).sort_values("rank_test_score", ascending=True).to_markdown())

export = True
if export:
    prediction = search.best_estimator_.predict(X_test)

    unique, counts = np.unique(prediction, return_counts=True)
    print(f"prediction stats: {dict(zip(unique, counts))}")

    out = []
    for i in range(len(prediction)):
        out.append([int(i), round(prediction[i])])

    filename = f"results/{search.best_score_:.3f}-{slugify(json.dumps({**search.best_params_}))}.csv"
    np.savetxt(
        filename,
        out,
        fmt=["%1.1f", "%1.14f"],
        delimiter=",",
        header="id,y",
        comments="",
    )
    print("done, saved " + filename)
