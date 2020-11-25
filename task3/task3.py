import numpy as np

from biosppy.signals.ecg import ecg
from lib.load import load_data

# just for development: limit amount of samples for quick iterations
X, X_test, y = load_data(limit=100)


features = []

for _sample in X:
    sample = _sample[~np.isnan(_sample)]
    res = ecg(sample, sampling_rate=300, show=False)

    heart_rate = res["heart_rate"]

    features.append([np.median(heart_rate), np.average(heart_rate), np.var(heart_rate)])

features = np.array(features)
print(f"computed features {features.shape}")

np.savetxt("features/features.csv", features, delimiter=",")
print("wrote features to features/features.csv")
