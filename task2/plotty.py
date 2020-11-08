from __future__ import print_function
import time
import numpy as np
import pandas as pd
# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from lib.load import load_data


X, X_test, y = load_data()
feat_cols = [ 'feature'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
test_df = pd.DataFrame(X_test,columns=feat_cols)
test_df['y'] = 3.0
test_df['label'] = test_df['y'].apply(lambda i: str(i))
# df = df[df["y"] != 1]
#
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
pca_test = pca.transform(X_test)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
test_df['pca-one'] = pca_test[:,0]
test_df['pca-two'] = pca_test[:,1]
test_df['pca-three'] = pca_test[:,2]



print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df[df["y"] == 1]["pca-one"],
    ys=df[df["y"] == 1]["pca-two"],
    zs=df[df["y"] == 1]["pca-three"],
    # c=df[df["y"] == 1]["y"],
    # cmap='tab10',
    label="1"
)
ax.scatter(
    xs=df[df["y"] == 2]["pca-one"],
    ys=df[df["y"] == 2]["pca-two"],
    zs=df[df["y"] == 2]["pca-three"],
    # c=df[df["y"] == 2]["y"],
    # cmap='tab10',
    label="2"
)
ax.scatter(
    xs=df[df["y"] == 0]["pca-one"],
    ys=df[df["y"] == 0]["pca-two"],
    zs=df[df["y"] == 0]["pca-three"],
    # c=df[df["y"] == 0]["y"],
    # cmap='tab10',
    label="0"
)
ax.scatter(
    xs=test_df["pca-one"],
    ys=test_df["pca-two"],
    zs=test_df["pca-three"],
    # c=df[df["y"] == 0]["y"],
    # cmap='tab10',
    label="test"
)
# ax.scatter(
#     xs=df[df["y"] == 2]["pca-one"],
#     ys=df[df["y"] == 2]["pca-two"],
#     zs=df[df["y"] == 2]["pca-three"],
#     c=df["y"],
#     # cmap='tab10',
#     label="1"
# )
# ax.scatter(
#     xs=df[df["y"] == 1]["pca-one"],
#     ys=df[df["y"] == 1]["pca-two"],
#     zs=df[df["y"] == 1]["pca-three"],
#     c=df["y"],
#     # cmap='tab10',
#     label="1"
# )
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.legend()
plt.show()


time_start = time.time()
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
df['tsne-2d-three'] = tsne_results[:,2]
plt.figure(figsize=(16,10))
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df[df["y"] == 1]["tsne-2d-one"],
    ys=df[df["y"] == 1]["tsne-2d-two"],
    zs=df[df["y"] == 1]["tsne-2d-three"],
    # c=df[df["y"] == 1]["y"],
    # cmap='tab10',
    label="1"
)
ax.scatter(
    xs=df[df["y"] == 2]["tsne-2d-one"],
    ys=df[df["y"] == 2]["tsne-2d-two"],
    zs=df[df["y"] == 2]["tsne-2d-three"],
    # c=df[df["y"] == 2]["y"],
    # cmap='tab10',
    label="2"
)
ax.scatter(
    xs=df[df["y"] == 0]["tsne-2d-one"],
    ys=df[df["y"] == 0]["tsne-2d-two"],
    zs=df[df["y"] == 0]["tsne-2d-three"],
    # c=df[df["y"] == 0]["y"],
    # cmap='tab10',
    label="0"
)
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two", z="tsne-2d-three",
#     hue="y",
#     palette=sns.color_palette("hls", 3),
#     data=df,
#     legend="full",
#     alpha=0.3
# )
plt.show()