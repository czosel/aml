import sklearn
from sklearn import svm
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.kernel_ridge import KernelRidge


def svr(X, X_test, y, params):
    C = params.get("C", 10)
    tol = params.get("svr_tol", 1)
    gamma = params.get("gamma", "scale")
    epsilon = params.get("epsilon", 0.1)
    kernel = params.get("kernel", "rbf")
    regr = svm.SVR(kernel=kernel, gamma=gamma, epsilon=epsilon, C=C, tol=tol)
    kf = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(regr, X, y, cv=kf)
    regr.fit(X, y)
    return regr.predict(X_test), scores


def kernel_ridge(X, X_test, y, params):
    alpha = params.get("alpha", 1)
    gamma = params.get("gamma", 0.1)
    kernel = params.get("kernel", "rbf")
    regr = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
    kf = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(regr, X, y, cv=kf)
    regr.fit(X, y)
    return regr.predict(X_test), score
