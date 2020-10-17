import numpy as np
import itertools
import json
import joblib
from pathlib import Path
import pickle
import urllib.request
from lib.prepare import prepare_data
from lib.svr import svr
import lib.meta as meta
from slugify import slugify


def load_csv(path):
    return np.genfromtxt(path, dtype=float, delimiter=",")


X_test_raw = load_csv("data/X_test.csv")
X_train_raw = load_csv("data/X_train.csv")
y_raw = load_csv("data/y_train.csv")

print("training data", X_train_raw.shape)
print("test data", X_test_raw.shape)


def run_all(config, prepare_config):
    results = {}

    prepare_keys, prepare_values = zip(*prepare_config.items())
    for prepare_bundle in itertools.product(*prepare_values):
        _prepare_config = dict(zip(prepare_keys, prepare_bundle))
        cache_key = f"joblib/prepare/{slugify(json.dumps(_prepare_config))}"

        if _prepare_config.get("load") and Path(cache_key).is_file():
            # print(f"Prepare config: {_prepare_config} (from cache)")
            X, X_test, y = joblib.load(cache_key)
        else:
            # print(f"Prepare config: {_prepare_config} (no cache)")
            X, X_test, y = prepare_data(X_train_raw, X_test_raw, y_raw, _prepare_config)
            if _prepare_config.get("save"):
                joblib.dump((X, X_test, y), cache_key)

        for algo, params in config.items():
            # see http://stephantul.github.io/python/2019/07/20/product-dict/
            keys, values = zip(*params.items())
            for bundle in itertools.product(*values):
                _params = dict(zip(keys, bundle))
                prediction, scores = globals()[algo](X, X_test, y, _params)
                # results[algo] = prediction, scores, y, X, train_prediction, X_test
                print(
                    f"{scores.mean():.2f}{highlight(scores.mean())} (+/- {scores.std():.2f}); algo={algo}, prepare_config={_prepare_config}, params={_params}"
                )

    return results


def select_algo(results):
    print("Baseline: ", max([results[algo][1] for algo, _ in results]))
    errors = np.array(
        [
            np.square(np.array(results[algo][2]) - np.array(results[algo][3]))
            for algo, _ in results
        ]
    )
    best = np.argmax(errors, axis=1)
    y = [results[algo][2] for algo, _ in results][0]
    X = [results[algo][3] for algo, _ in results][0]
    X_test = [results[algo][5] for algo, _ in results][0]
    train_pred = np.array([results[algo][4] for algo, _ in results])
    test_pred = np.array([results[algo][0] for algo, _ in results])
    return meta.create_best_of(X, best, y, train_pred, test_pred, X_test)


def highlight(score):
    if score > 0.6:
        return "***"
    if score > 0.55:
        return "** "
    if score > 0.5:
        return "*  "
    return "   "


def run_one(algo, select=True, clip=True, export=True, params={}):
    X, X_test, y = prepare_data(X_train_raw, X_test_raw, y_raw, select, clip)
    prediction, scores = globals()[algo](X, X_test, y, params)
    print(f"{algo}: {scores.mean():.2f} (+/- {scores.std():.2f})")

    if export:
        out = []
        for i in range(len(prediction)):
            out.append([int(i), prediction[i]])

        filename = f"{algo}-{slugify(json.dumps(params))}.csv"
        np.savetxt(
            filename,
            out,
            fmt=["%1.1f", "%1.14f"],
            delimiter=",",
            header="id,y",
            comments="",
        )
        print("done, saved " + filename)


# algorithms = ["xg"] # "lasso", "svr", "xg"
# algorithms = [ "lasso", "ridge", "svr"]
config = {
    # "lasso": {"alpha": [0.1, 0.25, 0.5, 0.75,1.0,1.5, 2.5, 4], "lasso_tol": [1e-3] },
    "svr": {
        "C": [50],
        "epsilon": [0.1],
        # "gamma": ["scale"],  # "auto", 1e-5, 1e-3, 0.1
        # "kernel": ["rbf"],  # "linear", "poly", "sigmoid"
    },
    # "kernel_ridge": {
    #   "alpha": [0.01, 0.03, 0.1, 0.3, 1],
    #   "gamma": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    #   "kernel": [ "laplacian", "rbf", ], #"linear", "poly", "polynomial", "rbf", "laplacian", "sigmoid", "cosine"],
    # },
}
# algorithms = [ "kernel_ridge"]
# algorithms = [ "shallow_net"]
# algorithms = [ "KNN"]
# algorithms = [ "GMR"]

params = {
    # "max_depth": [2,4,6]
}

results = run_all(
    config,
    {
        "load": [True],
        "save": [True],
        "n_features": [1, 10, 80],
        "direction": ["forward"],
    },
)
# run_one("svr", params={"C": 50, "epsilon": 0.01, "dir": "backward", "sel": "0.5/25"})
