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

fitting = Pipeline(
    [
        ("impute", sklearn.impute.SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        # ("PCA", PCA(n_components=100, whiten=True)),
        # ("KernelPCA", KernelPCA(n_components=100)),
        # (
        #     "target_svr",
        #     TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()),
        # ),
        # ("SVC", CustomClassifier()),
        ("SVC", SVC(cache_size=2000)),
        # ("NuSVC", NuSVC(class_weight="balanced", cache_size=2000)),
        # ("GaussianNB", GaussianNB()),  # 0.61
        # ("KNN", KNeighborsClassifier()),  # 0.55
        # (
        #     "kernel_ridge",
        #     TransformedTargetRegressor(
        #         regressor=KernelRidge(), transformer=StandardScaler()
        #     ),
        # ),
        # ("knn", KNeighborsRegressor()),
        # ("xgb", XGBClassifier(objective="multi:softmax")),
        # ("LGBM", LGBMClassifier(n_jobs=-1, objective="multiclassova", num_class=4))
        # ("LGBM", LGBMClassifier(n_jobs=-1, objective="multiclassova", num_class=4))
        # ("CAT", CatBoostClassifier(loss_function='MultiClass'))
        # ("deep", WideAndDeepNetClassifier())
    ],
    memory="cache",
)

param_distributions = {
    # "Custom__C": loguniform(1e-5, 1),
    # "LinearSVC__C": loguniform(1e-5, 1),
    # "LinearSVC__class_weight": [
    #     # "balanced",
    #     {
    #         0.0: 2,
    #         1.0: 1,
    #         2.0: 2,
    #     },
    # ],
    # "LinearSVC__loss": ["hinge", "squared_hinge"],
    # "PCA__n_components": stats.randint(100, 400),
    # "KernelPCA__n_components": stats.randint(100, 500),
    # "KernelPCA__kernel": ["linear", "poly", "rbf", "cosine", "sigmoid"],
    "SVC__C": loguniform(1e-3, 1e1),
    "SVC__gamma": ["scale", "auto"],
    "SVC__kernel": ["rbf", "poly", "linear", "sigmoid"],
    "SVC__class_weight": ["balanced", None],
    # "Stack__SVC1__C": loguniform(1e-5, 1e-3),
    # "Stack__SVC2__C": loguniform(1e-1, 10),
    # "NuSVC__nu": loguniform(1e-3, 0.3),
    # "NuSVC__kernel": ["linear"],
    # "GaussianNB__var_smoothing": loguniform(1e-11, 1e-3),
    # "KNN__n_neighbors": stats.randint(low=2, high=50),
    # "KNN__weights": ["uniform", "distance"],
    # "LogisticRegression__C": loguniform(1e-4, 1),
    # "MLP__hidden_layer_sizes": stats.randint(10, 200),
    # "MLP__alpha": loguniform(1e-6, 1),
    # "Quadratic__reg_param": loguniform(1e-6, 1),
    # "RandomForest__n_estimators": stats.randint(10, 100),
    # "Ridge__alpha": loguniform(1e-2, 100),
    # "svr__C": loguniform(0.1, 100),
    # "knn__n_neighbors": stats.randint(low=2, high=50),
    # "xgb__n_estimators": [4, 10, 100, 500, 1000],
    # "xgb__max_depth": [2, 4, 6, 10],
    # "xgb__booster": ["gbtree", "gblinear"],
    # "LGBM__num_leaves": [5, 10, 40, 80, 120, 200],
    # "LGBM__min_data_in_leaf": [10,100,1000,1000],
    # "LGBM__class_weight": [None],
    # "CAT__class_weights": [[6, 1, 6]], #[[1,1,1], [6, 1, 6]],
    # "CAT__depth": [8], #[4,6,8,10,12],
    # 'CAT__iterations': [50], #[10,50,100],
    #  'CAT__learning_rate': [0.01, 0.1, 1],
    #  'CAT__random_strength': [0.0001, 0.01, 10],
    #  'CAT__bagging_temperature': [0.0, 1.0],
    #  'CAT__border_count': [32, 255],
    #  'CAT__l2_leaf_reg':[2, 10, 30],
    # "target_svr__regressor__C": stats.expon(scale=100),
    # "target_svr__regressor__epsilon": stats.expon(),
    # "kernel_ridge__regressor__alpha": loguniform(1, 1e4),
    # "kernel_ridge__regressor__gamma": [0.1],
    # "deep__activation": ["tanh", "relu", "swish"],
    # "deep__nfirst": [32,64,128,264],
    # "deep__regularization": ["dropout"],
    # "deep__weights": [[6.0,1.0,6.0],[5.0,1.0,5.0],[7.0,1.0,7.0]],
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

export = False
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
