import numpy as np
import json
from slugify import slugify
from sklearn.svm import SVR
from sklearn.ensemble import IsolationForest
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
from xgboost import XGBRegressor
from lib.load import load_data
from sklearn.linear_model import BayesianRidge
from sklearn.utils.fixes import loguniform


X, X_test, y = load_data()

fitting = Pipeline(
    [
        ("scale", RobustScaler()),
        ("svr", SVR()),
        # (
        #     "target_svr",
        #     TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()),
        # ),
        # (
        #     "kernel_ridge",
        #     TransformedTargetRegressor(
        #         regressor=KernelRidge(), transformer=StandardScaler()
        #     ),
        # ),
        # ("knn", KNeighborsRegressor()),
        # ("xgb", XGBRegressor(objective="reg:squarederror")),
    ],
    memory="cache",
)

param_distributions = {
    "svr__C": loguniform(50, 200),
    "svr__epsilon": loguniform(1e-4, 1),
    # "knn__n_neighbors": stats.randint(low=2, high=50),
    # "xgb__n_estimators": stats.randint(low=50, high=300),
    # "xgb__max_depth": stats.randint(low=2, high=10),
    # "target_svr__regressor__C": stats.expon(scale=100),
    # "target_svr__regressor__epsilon": stats.expon(),
    # "kernel_ridge__regressor__alpha": loguniform(1, 1e4),
    # "kernel_ridge__regressor__gamma": [0.1],
}
search = RandomizedSearchCV(
    fitting,
    param_distributions,
    n_iter=2,
    n_jobs=2,
).fit(X, y)
print(f"CV score={search.best_score_:.3f}): {search.best_params_}")

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
