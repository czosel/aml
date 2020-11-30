import numpy as np
import pandas as pd
import json
from tabulate import tabulate
from slugify import slugify
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    FunctionTransformer,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import sklearn
from sklearn.pipeline import Pipeline
from lib.ThreeStepClassifier import ThreeStepClassifier
from lib.LSTMClassifier import LSTMClassifier
import itertools
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
import pandas as pd

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

# def load_data():

X, X_test, y = np.loadtxt("features/LGBMCache/train_X.csv", delimiter=","), np.loadtxt("features/LGBMCache/test_X.csv", delimiter=","), np.loadtxt("features/LGBMCache/train_y.csv", delimiter=",")

fitting = Pipeline(
    [
        ("scale", RobustScaler()),
        ("LGBM", LGBMClassifier(n_jobs=-1, num_class=4)),
        # ("SVC", ThreeStepClassifier(
        #     classifiers=[LGBMClassifier(n_jobs=-1, objective="binary"), LGBMClassifier(n_jobs=-1, objective="binary"),
        #                  LSTMClassifier(ncells=3, activation="tanh")])),

    ],
    memory="cache",
)

param_distributions = {
    "LGBM__num_leaves": [80, 120],
    "LGBM__min_data_in_leaf": [30, 50],
    "LGBM__objective": ["multiclass" ,"multiclassova"]

        # [LGBMClassifier(n_jobs=-1, objective="binary"),LSTMClassifier(ncells=3, activation="tanh"),LGBMClassifier(n_jobs=-1, objective="binary")],
        # [LGBMClassifier(n_jobs=-1, objective="binary"),LSTMClassifier(ncells=3, activation="relu"),LGBMClassifier(n_jobs=-1, objective="binary")],
        # [LGBMClassifier(n_jobs=-1, objective="binary"),LSTMClassifier(ncells=15, activation="tanh"),LGBMClassifier(n_jobs=-1, objective="binary")],
        # [LGBMClassifier(n_jobs=-1, objective="binary"),LSTMClassifier(ncells=15, activation="relu"),LGBMClassifier(n_jobs=-1, objective="binary")]
        # [LSTMClassifier(ncells=15, activation="tanh"), LGBMClassifier(n_jobs=-1, objective="binary"),
        #  LGBMClassifier(n_jobs=-1, objective="binary")],
        # [LSTMClassifier(ncells=3, activation="tanh"), LGBMClassifier(n_jobs=-1, objective="binary"),
        #  LGBMClassifier(n_jobs=-1, objective="binary")],
        # [LGBMClassifier(n_jobs=-1, objective="binary"),
        #  LGBMClassifier(n_jobs=-1, objective="binary"),LSTMClassifier(ncells=3, activation="tanh")],
        # [ LGBMClassifier(n_jobs=-1, objective="binary"),
        #  LGBMClassifier(n_jobs=-1, objective="binary"),LSTMClassifier(ncells=3, activation="relu")],
        # [ LGBMClassifier(n_jobs=-1, objective="binary"),
        #  LGBMClassifier(n_jobs=-1, objective="binary"),LSTMClassifier(ncells=15, activation="tanh")],
        # [LGBMClassifier(n_jobs=-1, objective="binary"),
        #  LGBMClassifier(n_jobs=-1, objective="binary"),LSTMClassifier(ncells=15, activation="relu")],

}

ppl = True
estm = None
if ppl:
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

    print(pd.DataFrame(search.cv_results_).sort_values("rank_test_score", ascending=True).to_markdown())
    estm = search.best_estimator_


else:
    estm = fitting


# print(classification_report(y, search.best_estimator_.predict(X)))

# estm.fit(X,y)
    # , plot_confusion_matrix

y_pred = cross_val_predict(estm, X, y, cv=5)
print(f"F1_MICRO score of current=", sklearn.metrics.f1_score(y,y_pred,average='micro'))
conf_mat = confusion_matrix(y, y_pred)
print("classification_report of best estimator:")
print(classification_report(y,y_pred))
plot_confusion_matrix(conf_mat,[0,1,2,3])
plt.show()

export = True
if export:

    y_pred = cross_val_predict(estm, X, y, cv=5, method='predict_proba')
    filename = f"features/LGBMCache/y_feat.csv"
    print(y_pred.shape)
    np.savetxt(
        filename,
        y_pred,
        delimiter=",",
    )
    filename = f"features/LGBMCache/y_test.csv"

    estm.fit(X, y)
    prediction = estm.predict_proba(X_test)
    unique, counts = np.unique(prediction, return_counts=True)
    print(f"prediction stats: {dict(zip(unique, counts))}")
    print(prediction.shape)
    np.savetxt(
        filename,
        prediction,
        delimiter=",",
    )
    print("done, saved " + filename)
