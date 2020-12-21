import numpy as np
import mne
import matplotlib.pyplot as plt

from lib.load import load_data

# just for development: limit amount of samples for quick iterations
# X, X_test, y = load_data(limit=5)
X, X_test, y = load_data()

info = mne.create_info(["eeg1", "eeg2"], 128, ["eeg", "eeg"])
info2 = mne.create_info(["emg1"], 128, ["emg"])

beta = list(range(27, 48))
alpha = list(range(17, 27))
delta = list(range(0, 7))
theta = list(range(7, 17))

ranges = [beta, alpha, delta, theta]


def process(X):
    X_eeg1, X_eeg2, X_emg = X
    features = []
    for epoch in range(len(X_eeg1)):
        raw = mne.io.RawArray(np.array([X_eeg1[epoch], X_eeg2[epoch]]), info)
        psds, freqs = mne.time_frequency.psd_welch(
            raw, fmin=0.5, fmax=24, n_overlap=128
        )

        sums1 = [np.sum(psds[0]) for rng in ranges]
        sums2 = [np.sum(psds[1]) for rng in ranges]

        raw2 = mne.io.RawArray(np.array([X_emg[epoch]]), info2)
        psds2, freqs2 = mne.time_frequency.psd_welch(raw2, fmax=30, picks=["emg1"])

        emg_sum = np.sum(psds2[0])

        features.append(np.log([emg_sum] + sums1 + sums2))

    return features


np.savetxt("features/cache/train_X.csv", process(X), delimiter=",")
np.savetxt("features/cache/train_y.csv", y, delimiter=",")
np.savetxt("features/cache/test_X.csv", process(X_test), delimiter=",")
print("wrote features")
