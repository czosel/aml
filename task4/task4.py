import numpy as np
import mne
import matplotlib.pyplot as plt

from lib.load import load_data

# just for development: limit amount of samples for quick iterations
# X, X_test, y = load_data(limit=5)
X, X_test, y = load_data()

info = mne.create_info(["eeg1", "eeg2", "emg"], 128, ["eeg", "eeg", "emg"])


def process(X):
    X_eeg1, X_eeg2, X_emg = X
    features = []
    for epoch in range(len(X_eeg1)):
        raw = mne.io.RawArray(
            np.array([X_eeg1[epoch], X_eeg2[epoch], X_emg[epoch]]), info
        )

        # scalings = {"eeg": 1e-3, "emg": 5e-5}
        # raw.plot(n_channels=3, scalings=scalings, show=True, block=True)

        # psd
        # raw.plot_psd()
        psds, freqs = mne.time_frequency.psd_welch(raw, fmax=24)
        psds_mean = psds.mean(0)
        psds_std = psds.std(0)
        # plt.plot(freqs, psds_mean, color="k")
        # plt.fill_between(
        #     freqs, psds_mean - psds_std, psds_mean + psds_std, color="k", alpha=0.5
        # )
        # plt.show()

        features.append(psds_mean + psds_std)
    return features


print(process(X).shape)
np.savetxt("features/cache/train_X.csv", process(X), delimiter=",")
np.savetxt("features/cache/train_y.csv", y, delimiter=",")
np.savetxt("features/cache/test_X.csv", process(X_test), delimiter=",")
print("wrote features")
