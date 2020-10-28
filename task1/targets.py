import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


def load_csv(path):
    return np.genfromtxt(path, dtype=float, delimiter=",")


y_raw = load_csv("data/y_train.csv")
y = np.ravel(np.delete(np.delete(y_raw, 0, 0), 0, 1))

plt.hist(y, 30)
plt.show()
