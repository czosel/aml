import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from lib.load import load_data
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import collections

X, X_test, y = load_data()

scaler = RobustScaler()
X = scaler.fit_transform(X)

clusters = 3
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
pred = kmeans.predict(X)


pca = PCA(n_components=3)
pca_result = pca.fit_transform(X)
# pca_test = pca.transform(X_test)
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.set_title("K-means")
for c in range(clusters):
    indices = pred == c
    cnts = collections.Counter(y[indices])
    print(cnts)
    print("Percentage of 1s: ", cnts[1.0] / sum(cnts.values()))
    fig1, ax2 = plt.subplots()
    ax2.set_title("Cluster " +str(c))
    ax2.pie(collections.Counter(y[indices]).values())
    ax.scatter(
        xs=pca_result[indices,0],
        ys=pca_result[indices,1],
        zs=pca_result[indices,2],
        # c=df[df["y"] == 1]["y"],
        # cmap='tab10',
        label=str(c)+"  "+str(cnts[1.0] / sum(cnts.values()))
    )
ax.legend()
plt.show()

mx = GaussianMixture(n_components=clusters)
mx.fit(X)
# pred = np.round(mx.predict(X))
pred = np.round(mx.predict(X))


pca = PCA(n_components=3)
pca_result = pca.fit_transform(X)
# pca_test = pca.transform(X_test)
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.set_title("Gaussian mixture fit")
for c in range(clusters):
    indices = pred == c
    cnts = collections.Counter(y[indices])
    print(cnts)
    print("Percentage of 1s: ", cnts[1.0] / sum(cnts.values()))
    fig1, ax2 = plt.subplots()
    ax2.set_title("Cluster " +str(c))
    ax2.pie(collections.Counter(y[indices]).values())
    ax.scatter(
        xs=pca_result[indices,0],
        ys=pca_result[indices,1],
        zs=pca_result[indices,2],
        # c=df[df["y"] == 1]["y"],
        # cmap='tab10',
        label=str(c)+"  "+str(cnts[1.0] / sum(cnts.values()))
    )
ax.legend()
plt.show()

