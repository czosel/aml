import numpy as np
import itertools
import pickle
import urllib.request
from lib.prepare import prepare_data
from lib.svr import svr
import lib.meta as meta


def load_csv(path):
    return np.genfromtxt(path, dtype=float, delimiter=",")


X_test_raw = load_csv("data/X_test.csv")
X_train_raw = load_csv("data/X_train.csv")
y_raw = load_csv("data/y_train.csv")

print("training data", X_train_raw.shape)
print("test data", X_test_raw.shape)


if "prepare_cache" not in globals():
    prepare_cache = {}


def run_all(config, prepare_config):
    results = {}

    for _select in prepare_config.pop("select", [True]):
        for _clip in prepare_config.pop("clip", [True]):
            if prepare_config.get("cache") and prepare_cache.get(
                pickle.dumps(prepare_config)
            ):
                print(f"Select: {_select}, Clip: {_clip} (from cache)")
                X, X_test, y = prepare_cache[pickle.dumps(prepare_config)]
            else:
                print(f"Select: {_select}, Clip: {_clip} (no cache)")
                X, X_test, y = prepare_data(
                    X_train_raw, X_test_raw, y_raw, _select, _clip
                )
                prepare_cache[pickle.dumps(prepare_config)] = (X, X_test, y)

            for algo, params in config.items():
                print(f"## {algo} ##")
                # see http://stephantul.github.io/python/2019/07/20/product-dict/
                keys, values = zip(*params.items())
                for bundle in itertools.product(*values):
                    _params = dict(zip(keys, bundle))
                    prediction, scores, train_prediction = globals()[algo](X, X_test, y, _params)
                    results[algo] = prediction, scores, y, X, train_prediction, X_test
                    print(
                        f"{scores.mean():.2f}{highlight(scores.mean())} (+/- {scores.std():.2f}); algo={algo}, params={_params}"
                    )

    return results

def select_algo(results):
    print("Baseline: ", max([results[algo][1] for algo, _ in results]))
    errors = np.array([np.square(np.array(results[algo][2]) - np.array(results[algo][3])) for algo, _ in results])
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


def run_one(algo, select=True, clip=True, export=True, **kwargs):
    X, X_test, y = prepare_data(X_train_raw, X_test_raw, y_raw, select, clip)
    prediction, scores = globals()[algo](X, X_test, y, **kwargs)
    print(f"{algo}: {scores.mean():.2f} (+/- {scores.std():.2f})")

    if export:
        out = []
        for i in range(len(prediction)):
            out.append([int(i), prediction[i]])

        filename = f"{algo}-select-{select}-clip-{clip}.csv"
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
        "C": [10, 20, 30, 50, 70, 100, 200],
        "epsilon": [0.01, 0.1, 1],
        "gamma": ["scale"],  # "auto", 1e-5, 1e-3, 0.1
        "kernel": ["rbf"],  # "linear", "poly", "sigmoid"
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

results = run_all(config, {"cache": False, "select": [True], "clip": [True]})
# run_one("svr")
