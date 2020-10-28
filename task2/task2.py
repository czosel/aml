import numpy as np
import pandas as pd
import json
from tabulate import tabulate
from slugify import slugify
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.neighbors import KNeighborsClassifier
from sklego.mixture import GMMOutlierDetector, BayesianGMMOutlierDetector
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
from lib.load import load_data
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.utils.fixes import loguniform

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


X, X_test, y = load_data()

fitting = Pipeline(
    [
        ("scale", RobustScaler()),
        # (
        #     "LinearSVC",
        #     LinearSVC(multi_class="crammer_singer", class_weight="balanced", tol=1),
        # ),
        ("SVC", SVC(class_weight="balanced")),
        # ("GaussianNB", GaussianNB()),  # 0.61
        # ("KNN", KNeighborsClassifier()),  # 0.55
        # (
        #     "LogisticRegression",
        #     LogisticRegression(class_weight="balanced", tol=1),
        # ), # 0.698
        # ("MLP", MLPClassifier()),  # 0.63
        # ("Quadratic", QuadraticDiscriminantAnalysis()), # 0.36
        # ("RandomForest", RandomForestClassifier()),  # 0.515
        # ("Ridge", RidgeClassifier(class_weight="balanced")),  # 0.657
    ],
    memory="cache",
)

param_distributions = {
    # "LinearSVC__C": loguniform(1e-5, 1),
    "SVC__C": loguniform(1e-5, 1),
    "SVC__gamma": ["scale", "auto"],
    # "SVC__kernel": ["rbf"]
    # "GaussianNB__var_smoothing": loguniform(1e-11, 1e-3),
    # "KNN__n_neighbors": stats.randint(low=2, high=50),
    # "KNN__weights": ["uniform", "distance"],
    # "LogisticRegression__C": loguniform(1e-4, 1),
    # "MLP__hidden_layer_sizes": stats.randint(10, 200),
    # "MLP__alpha": loguniform(1e-6, 1),
    # "Quadratic__reg_param": loguniform(1e-6, 1),
    # "RandomForest__n_estimators": stats.randint(10, 100),
    # "Ridge__alpha": loguniform(1e-2, 100)
}
search = RandomizedSearchCV(
    fitting,
    param_distributions,
    scoring="balanced_accuracy",
    n_iter=10,
    n_jobs=2,
    verbose=2,
).fit(X, y)
print(f"BMAC score={search.best_score_:.3f}): {search.best_params_}")
frame = pd.DataFrame.from_dict(search.cv_results_)
print(
    tabulate(
        frame[["params", "mean_test_score", "std_test_score", "rank_test_score"]],
        headers="keys",
    )
)

export = False
if export:
    prediction = search.best_estimator_.predict(X_test)
    out = []
    for i in range(len(prediction)):
        out.append([int(i), round(prediction[i])])

    filename = f"results/{slugify(json.dumps({**search.best_params_}))}.csv"
    np.savetxt(
        filename,
        out,
        fmt=["%1.1f", "%1.14f"],
        delimiter=",",
        header="id,y",
        comments="",
    )
    print("done, saved " + filename)
