import numpy as np
from scipy.fft import fft
from scipy.stats import skew
from itertools import chain
import pandas as pd

from biosppy.signals.ecg import ecg
from pyhrv import hrv, time_domain, frequency_domain

from lib.load import load_data
import matplotlib.pyplot as plt

# just for development: limit amount of samples for quick iterations
X, X_test, y = load_data(limit=None)


def calc_median(series):
    return np.array([np.median(series[:, k]) for k in range(180)]), np.array([np.mean(series[:, k]) for k in range(180)]), np.array([np.std(series[:, k]) for k in range(180)])


def stats(arr):
    return [np.median(arr), np.var(arr), skew(arr)]


def build_features(*features):
    return list(chain(*[stats(f) for f in features]))


def find_minimum(signal, peak_index, max_span=50, direction="left"):
    i = peak_index
    next_i = i - 1 if direction == "left" else i + 1
    count = 0

    while signal[next_i] < signal[i] and count <= max_span:
        i = next_i
        next_i = next_i - 1 if direction == "left" else next_i + 1
        count = count + 1

    if count == max_span:
        search = (
            signal[peak_index - max_span : peak_index]
            if direction == "left"
            else signal[peak_index : peak_index + max_span]
        )
        print(f"did not find local mininimum at {peak_index} in {search}")

    return i


def process(X):
    features = []
    for i in range(len(X)):
        _sample = X[i]
        print(f"sample {i}: Class {y[i]}")
        sample = _sample[~np.isnan(_sample)]

        res = ecg(signal=sample, sampling_rate=300, show=False)

        median, mean, std = calc_median(res["templates"])
        # if (np.argmin(median) < 60) and not 0.7*np.max(median) > abs(np.min(median)):
        if (
            not np.max(median[55:65]) == np.max(median)
            or (np.max(median) < -0.8 * np.min(median))
            or (
                not 0.75 * np.max(median) > -np.min(median)
                and (
                    np.argmin(median) < 60
                    and (
                        np.min(median[:60]) < 1.5 * min(median[60:])
                        or np.min(median[60:]) < 1.5 * min(median[:60])
                    )
                )
            )
        ):
            # and ((np.min(median) < 1.2 * np.min(
            #     median[[i for i in range(len(median)) if i != np.argmin(median)]])) or np.max(median[45:48]) > -np.min(median[65:75])):
            # if np.min(median[45:55]) < np.min(median[0:45]) and np.min(median[65:80]) < np.min(median[80:]) and np.max(median[55:65]) == np.max(median):
            # if np.max(median) < abs(np.min(median)) and np.min(median[50:55]) < np.min(median[60:65]):
            # if abs(np.mean(median)) > abs(np.median(median)):
            res = ecg(-sample, sampling_rate=300, show=False)

            median, mean, std  = calc_median(res["templates"])


        std =  median.std()
        vec = []
        for i in range(len(mean)):
            vec.append(np.log(mean[i]/median[i]))
        median = (median) / std
        mean = (mean) / std
        for m in median:
            vec.append(m)
        for m in mean:
            vec.append(m)
        print(np.array(vec).shape)
        features.append(np.array(vec))

    features = np.array(features, dtype=np.float)


    print(f"computed features {features.shape}")
    return features


# scatter = plt.scatter(features[:, 0], features[:, 2], c=y, label=y)
# plt.legend(*scatter.legend_elements())
# plt.xlabel("median heart rate")
# plt.ylabel("heart rate variance")
# plt.show()


# tpls0 = np.swapaxes(np.array(tpls0), 0, 1)
# tpls1 = np.swapaxes(np.array(tpls1), 0, 1)
# tpls2 = np.swapaxes(np.array(tpls2), 0, 1)
# tpls3 = np.swapaxes(np.array(tpls3), 0, 1)
# print(np.array(tpls0).shape)
# plt.plot(tpls0)
# plt.show()
# plt.plot(tpls1)
# plt.show()
# plt.plot(tpls2)
# plt.show()
# plt.plot(tpls3)
# plt.show()

np.savetxt("features/LGBMCache/train_X.csv", process(X), delimiter=",")
np.savetxt("features/LGBMCache/train_y.csv", y, delimiter=",")
np.savetxt("features/LGBMCache/test_X.csv", process(X_test), delimiter=",")

# process(X).to_csv("features/LGBMCache/train_X.csv", index=False)
# np.savetxt("features/LGBMCache/train_y.csv", y, delimiter=",")
# process(X).to_csv("features/LGBMCache/test_X.csv", index=False)
print("wrote features")
