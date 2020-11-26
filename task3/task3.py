import numpy as np

from biosppy.signals.ecg import ecg
from lib.load import load_data
import matplotlib.pyplot as plt

# just for development: limit amount of samples for quick iterations
X, X_test, y = load_data(limit=20000)


features = []

plot = False
print(y[0:40])
tpls0 = []
tpls1 = []
tpls2 = []
tpls3 = []


def calc_median(series):
    return np.array([np.median(series[:, k]) for k in range(180)])


for i in range(len(X)):
    _sample = X[i]
    sample = _sample[~np.isnan(_sample)]
    res = ecg(sample, sampling_rate=300, show=False)

    median = calc_median(res["templates"])
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

        median = calc_median(res["templates"])
        # neg = True

        # res["templates"][j] = (res["templates"][j]-mean)/std

    median = (median) / median.std()
    if i < 40 and plot:
        # plt.plot(res["templates"][j])
        plt.title(y[i])
        plt.plot(median)
        plt.show()

    # if not neg:
    if y[i] == 0:
        tpls0.append(median)
    if y[i] == 1:
        tpls1.append(median)
    if y[i] == 2:
        tpls2.append(median)
    if y[i] == 3:
        tpls3.append(median)

    heart_rate = res["heart_rate"]

    # print(templates.shape, median.shape)
    features.append([np.median(heart_rate), np.average(heart_rate), np.var(heart_rate)])

tpls0 = np.swapaxes(np.array(tpls0), 0, 1)
tpls1 = np.swapaxes(np.array(tpls1), 0, 1)
tpls2 = np.swapaxes(np.array(tpls2), 0, 1)
tpls3 = np.swapaxes(np.array(tpls3), 0, 1)
print(np.array(tpls0).shape)
plt.plot(tpls0)
plt.show()
plt.plot(tpls1)
plt.show()
plt.plot(tpls2)
plt.show()
plt.plot(tpls3)
plt.show()

features = np.array(features)
print(f"computed features {features.shape}")

np.savetxt("features/features.csv", features, delimiter=",")
print("wrote features to features/features.csv")
