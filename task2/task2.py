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
from sklearn.decomposition import PCA, KernelPCA

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

X, X_test, y = load_data()


class CustomClassifier(BaseEstimator, ClassifierMixin):
    """
    Dummy implementation of custom classifier.

    See https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    """

    def __init__(self, C=1):
        self.C = C
        self.model = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.model = LinearSVC(C=self.C)
        self.model.fit(X, y)
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self.model)

        # Input validation
        X = check_array(X)

        return self.model.predict(X)


fitting = Pipeline(
    [
        ("scale", RobustScaler()),
        # ("PCA", PCA(n_components=100, whiten=True)),
        # ("KernelPCA", KernelPCA(n_components=100)),
        # (
        #     "LinearSVC",
        #     LinearSVC(multi_class="crammer_singer", tol=0.1),
        # ),
        ("Custom", CustomClassifier()),
        # ("SVC", SVC(class_weight="balanced", cache_size=2000)),
        # ("NuSVC", NuSVC(class_weight="balanced", cache_size=2000)),
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
        # (  # BMAC score=0.613): {'LogisticRegression__C': 0.00177, 'Ridge__alpha': 0.054, 'SVC__C': 16.04}
        #     "Stack",
        #     StackingClassifier(
        #         estimators=[
        #             ("SVC1", SVC(kernel="linear", class_weight="balanced")),
        #             ("SVC2", SVC(kernel="rbf", class_weight="balanced")),
        #         ],
        #         final_estimator=LogisticRegression(tol=1),
        #     ),
        # ),
    ],
    memory="cache",
)

param_distributions = {
    "Custom__C": loguniform(1e-5, 1),
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
}
search = RandomizedSearchCV(
    fitting,
    param_distributions,
    scoring="balanced_accuracy",
    n_iter=5,
    n_jobs=4,
    verbose=3,
    cv=5,
).fit(X, y)
print(f"BMAC score={search.best_score_:.3f}): {search.best_params_}")
frame = pd.DataFrame.from_dict(search.cv_results_)
print(
    tabulate(
        frame[["params", "mean_test_score", "std_test_score", "rank_test_score"]],
        headers="keys",
    )
)

print("classification_report of best estimator:")
print(classification_report(y, search.best_estimator_.predict(X)))

print("confusion matrix:")
print(confusion_matrix(y, search.best_estimator_.predict(X)))

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
