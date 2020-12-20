import numpy as np
import pandas as pd
import json
from tabulate import tabulate
from slugify import slugify
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import sklearn
from sklearn import metrics
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    FunctionTransformer,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
import itertools
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
# from lib.ShallowNetClassifier import ShallowNetClassifier
# from lib.DeepNetClassifier import DeepNetClassifier
# from lib.WideAndDeepNetClassifier import WideAndDeepNetClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.utils.fixes import loguniform
from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA, KernelPCA

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from matplotlib import pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def windowed(X, n_subjects, winsize_forward=5, winsize_backward=5):
    l = int(len(X) / 3)
    X1, X2, X3 = X[0:l], X[l : 2 * l], X[2 * l :]
    ret = []
    for X_ in [X1, X2, X3]:
        # repeat first and last samples such that every sample has all needed window
        padded = np.concatenate(
            (
                np.tile(X_[0], (5, 1)),
                X_,
                np.tile(X_[-1], (5, 1)),
            )
        )
        # flatten previous and following samples along with current sample
        ret.append(
            np.array(
                [
                    np.concatenate(padded[i - winsize_backward : i + winsize_forward])
                    for i in range(winsize_backward, len(X_) + winsize_forward)
                ]
            )
        )
    return np.concatenate(ret)


train_X = windowed(
    np.loadtxt("features/cache/train_X.csv", delimiter=","), n_subjects=3
)
train_y = np.loadtxt("features/cache/train_y.csv", delimiter=",")
test_X = windowed(np.loadtxt("features/cache/test_X.csv", delimiter=","), n_subjects=2)

fold_indices = tuple(
    [
        (
            np.arange(0, 2*int(len(train_X) / 3)),
            np.arange(2 * int(len(train_X) / 3), 3 * int(len(train_X) / 3)),
        ),
        (
            np.arange(int(len(train_X) / 3), 3 * int(len(train_X) / 3)),
            np.arange(0, int(len(train_X) / 3)),
        ),
        (
            np.concatenate((range(0,int(len(train_X) / 3)), range(2*int(len(train_X) / 3),3*int(len(train_X) / 3)))),
            np.arange(int(len(train_X) / 3), 2*int(len(train_X) / 3)),
        )
    ]
)

print(train_X.shape, train_y.shape)

fitting = Pipeline(
    [
        ("scale", RobustScaler()),
        ("PCA", PCA(n_components=100, whiten=True)),
        # ("KMeans", MiniBatchKMeans(n_clusters=100)),
        # ("KernelPCA", KernelPCA(n_components=100)),
        # (
        #     "target_svr",
        #     TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()),
        # ),
        # ("Custom", CustomClassifier()),
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
        # ("LGBM", LGBMClassifier(n_jobs=-1, objective="multiclassova", num_class=3))
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
    "SVC__kernel": ["rbf", "linear", "poly"],
    "SVC__class_weight": [None, "balanced"],
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
    # "LGBM__num_leaves": [80],#[5, 10, 40, 80, 120, 200],
    # "LGBM__min_data_in_leaf": [100],
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

ppl = True
estm = None
if ppl:
    search = RandomizedSearchCV(
        fitting,
        param_distributions,
        cv=fold_indices,
        scoring="balanced_accuracy",
        n_iter=20,
        n_jobs=1,
        verbose=2,
    ).fit(train_X, train_y)
    print(f"F1_MICRO score={search.best_score_:.3f}): {search.best_params_}")
    frame = pd.DataFrame.from_dict(search.cv_results_)
    print(
        tabulate(
            frame[["params", "mean_test_score", "std_test_score", "rank_test_score"]],
            headers="keys",
        )
    )

    print(pd.DataFrame(search.cv_results_).sort_values("rank_test_score", ascending=True).to_markdown())
    estm = search.best_estimator_


else:
    estm = fitting


# print(classification_report(y, search.best_estimator_.predict(X)))

# estm.fit(X,y)
    # , plot_confusion_matrix


y_pred = cross_val_predict(estm, train_X, train_y, cv=fold_indices)
print(f"F1_MICRO score of current=", metrics.balanced_accuracy_score(train_y,y_pred))
conf_mat = confusion_matrix(train_y, y_pred)
print("classification_report of best estimator:")
print(classification_report(train_y,y_pred))
plot_confusion_matrix(conf_mat,[0,1,2,3])
plt.show()

# print("classification_report of best estimator:")
# print(classification_report(train_y, search.best_estimator_.predict(train_X)))
# print(
#     pd.DataFrame(search.cv_results_)
#     .sort_values("rank_test_score", ascending=True)
#     .to_markdown()
# )
plt.plot(train_y[250:750], color="green")
plt.plot(y_pred[250:750], color="orange")
plt.show()

export = True
if export:
    prediction = search.best_estimator_.predict(test_X)

    unique, counts = np.unique(prediction, return_counts=True)
    print(f"prediction stats: {dict(zip(unique, counts))}")

    out = []
    for i in range(len(prediction)):
        out.append([int(i), int(prediction[i])])

    filename = f"results/{search.best_score_:.3f}-{slugify(json.dumps({**search.best_params_}))}.csv"
    np.savetxt(
        filename,
        out,
        fmt=["%1.1f", "%1.14f"],
        delimiter=",",
        header="Id,y",
        comments="",
    )
    print("done, saved " + filename)
