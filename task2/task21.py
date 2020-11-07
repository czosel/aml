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
from lib.ShallowNetClassifier import ShallowNetClassifier
from lib.DeepNetClassifier import DeepNetClassifier
from lib.WideAndDeepNetClassifier import WideAndDeepNetClassifier
from lib.OVAClassifier import OVAClassifier
from lib.OVOClassifier import OVOClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.utils.fixes import loguniform
from catboost import CatBoostClassifier
from  sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA, KernelPCA

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

X, X_test, y = load_data()

fitting = Pipeline(
    [
        ("scale", RobustScaler()),
        ("ova", OVAClassifier())
        # ("ovo", OVOClassifier())
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
        # "SVC__C": loguniform(1e-3, 1e1),
        # "SVC__gamma": ["scale", "auto"],
        # "SVC__kernel": ["rbf"],
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
        # "xgb__scale_pos_weight": [
        #     zeros*1.0/ones],
        # "LGBM__num_leaves": [5, 10, 40, 80, 120, 200],
        # "LGBM__min_data_in_leaf": [10,100,1000,1000],
        # "LGBM__is_unbalance": [True],
        # "CAT__class_weights": [[1.0, zeros*1.0/ones]], #[[1,1,1], [6, 1, 6]],
        # "CAT__depth": [4,6,8,10,12],
        # 'CAT__iterations': [10,35],
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
        # "deep__regularization": ["dropout", None],
        # "deep__pos_weight": [{0: ones/(zeros+ones) ,
# 1: zeros/(ones+zeros)},{0: ones/zeros ,
# 1: zeros/ones}],
        # "deep__pos_weight": [{0: (1 / zeros)*(zeros+ones)/2.0 ,
# 1: (1 / ones)*(ones+zeros)/2.0}],
#         "ovo__c1": [LinearSVC(class_weight='balanced', C=0.0020053280137711875,  loss="hinge")],
#         "ovo__c2": [SVC(class_weight="balanced", C=1.2781191419905644, kernel="rbf")],
#         # "ova__c2": [LinearSVC(class_weight="balanced", loss="squared_hinge", C=0.000243486)],
#         "ovo__c3": [LinearSVC(class_weight="balanced", loss="squared_hinge", C=0.0005165696988129459)],

    # OVA
            "ova__c1": [LinearSVC(class_weight={0:2,1:1}, C=0.000891058,  loss="squared_hinge")],
            # "ova__c2": [SVC(class_weight="balanced", C=1.1547940567988086, gamma='scale', kernel="rbf")],
            "ova__c2": [LinearSVC(class_weight={0:1,1:20}, loss="squared_hinge", C=0.000243486)],
            "ova__c3": [LinearSVC(class_weight={0:2,1:1}, loss="squared_hinge", C=0.000243486)],
            "ova__c_tie": [OVOClassifier(c1=LinearSVC(class_weight='balanced', C=0.0020053280137711875,  loss="hinge"), c2=SVC(class_weight="balanced", C=1.2781191419905644, kernel="rbf"), c3=LinearSVC(class_weight="balanced", loss="squared_hinge", C=0.0005165696988129459))],
    # "ova__c_tie": [LinearSVC(class_weight="balanced", C=0.0001, loss="hinge")],
    # "ova__c_tie": [SVC(class_weight="balanced", C=0.01, gamma='auto', kernel="rbf")],

}

search = RandomizedSearchCV(
    fitting,
    param_distributions,
    scoring="balanced_accuracy",
    n_iter=20,
    n_jobs=1,
    verbose=0,
).fit(X, y)
print(f"BMAC score={search.best_score_:.3f}): {search.best_params_}")
frame = pd.DataFrame.from_dict(search.cv_results_).sort_values("rank_test_score", ascending=True)
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
