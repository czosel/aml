import numpy as np
from scipy.fft import fft
from scipy.stats import skew
from itertools import chain

from biosppy.signals.ecg import ecg
from pyhrv import hrv, time_domain, frequency_domain

from lib.load import load_data
import matplotlib.pyplot as plt

# just for development: limit amount of samples for quick iterations
X, X_test, y = load_data()


def calc_median(series):
    return np.array([np.median(series[:, k]) for k in range(180)])


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

    plot = False
    print(y[0:40])
    tpls0 = []
    tpls1 = []
    tpls2 = []
    tpls3 = []

    shp = None
    for i in range(len(X)):
        _sample = X[i]
        print(f"sample {i}: Class {y[i]}")
        sample = _sample[~np.isnan(_sample)]

        res = ecg(signal=sample, sampling_rate=300, show=False)

        # FT
        # N = len(sample) / 2
        # T = 1.0 / 300.0

        # xf = np.linspace(0.0, 1.0 / (2 * T), int(N // 2))
        # yf = fft(res["filtered"])
        # plt.plot(xf, 2.0 / N * np.abs(yf[0 : int(N // 2)]))
        # plt.show()

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

        # beat characterization
        heart_rate = res["heart_rate"]

        filtered = res["filtered"]
        rpeaks = res["rpeaks"]
        # peaks in seconds, required by pyhrv
        rpeaks_s = res["ts"][rpeaks]
        qpeaks = np.array([find_minimum(filtered, r) for r in rpeaks])
        speaks = np.array(
            [find_minimum(filtered, r, direction="right") for r in rpeaks]
        )

        r_amplitude = filtered[rpeaks]
        q_amplitude = filtered[qpeaks]
        s_amplitude = filtered[speaks]

        qrs_duration = speaks - qpeaks

        # hrv_res = hrv(
        #    rpeaks=rpeaks_s,
        #    plot_tachogram=False,
        #    kwargs_ar={"order": 8},
        #    show=False,
        # )
        nni = time_domain.nni_parameters(rpeaks=rpeaks_s)
        nni_diff = time_domain.nni_differences_parameters(rpeaks=rpeaks_s)
        sdnn = time_domain.sdnn(rpeaks=rpeaks_s)
        sdsd = time_domain.sdsd(rpeaks=rpeaks_s)
        tri_index = time_domain.triangular_index(rpeaks=rpeaks_s, plot=False)

        welch_psd = frequency_domain.welch_psd(rpeaks=rpeaks_s, mode="dev")[0]
        # print(templates.shape, median.shape)
        features.append(
            build_features(q_amplitude, r_amplitude, s_amplitude, qrs_duration)
            + [
                nni["nni_mean"],
                nni["nni_min"],
                nni["nni_max"],
                nni_diff["nni_diff_mean"],
                nni_diff["nni_diff_min"],
                nni_diff["nni_diff_max"],
                sdnn["sdnn"],
                sdsd["sdsd"],
                tri_index["tri_index"],
                welch_psd["fft_ratio"],
            ]
            + list(welch_psd["fft_peak"] + welch_psd["fft_abs"] + welch_psd["fft_norm"])
        )
        # print(templates.shape, median.shape)

    features = np.array(features)
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


np.savetxt("features/cache/train_X.csv", process(X), delimiter=",")
np.savetxt("features/cache/train_y.csv", y, delimiter=",")
np.savetxt("features/cache/test_X.csv", process(X_test), delimiter=",")
print("wrote features")
