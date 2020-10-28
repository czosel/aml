import xgboost as xgb
import numpy as np
import sklearn
from sklearn.model_selection import KFold


def create_best_of(X, best, y, train_pred, test_pred, X_test):
    kf = KFold(n_splits=5, shuffle=True)
    scores = []
    for train, test in kf.split(X, y):
        regr = xgb.XGBClassifier(max_depth=3)
        regr.fit(X[train], best[train])
        # predictor = np.argmax(np.array(regr.predict(X[test])), axis=1)
        predictor = regr.predict(X[test])
        pred = [train_pred[predictor[i], test[i]] for i in range(len(test))]
        scores.append(sklearn.metrics.py, r2_score(test_pred, y[test]))

    print("CV scores: ", scores, mean(scores), std(scores))
    regr = xgb.XGBClassifier(max_depth=3)
    regr.fit(X, y)
    predictor = regr.predict(X_test)
    final = [test_pred[predictor[i], i] for i in range(len(test_pred[0, :]))]
    return final
