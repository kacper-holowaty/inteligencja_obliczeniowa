# PCA

from sklearn import datasets
from sklearn.decomposition import PCA 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris() 
X = pd.DataFrame(iris.data, columns=iris.feature_names) 
y = pd.Series(iris.target, name='FlowerType') 

pca_iris = PCA(n_components=3).fit(X) 
print(pca_iris) 
print(pca_iris.explained_variance_ratio_) 

res = pca_iris.explained_variance_ratio_.cumsum()
print(res)

components_to_keep = np.argmax(res >= 0.95) + 1
print("Liczba komponentów, które należy zachować: ", components_to_keep)


def wykres_3d(X, y):
    np.random.seed(5)
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()

    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])


    plt.cla()
    pca_iris = PCA(n_components=3).fit(X)
    X = pca_iris.transform(X)

    for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
        ax.text3D(
            X[y == label, 0].mean(),
            X[y == label, 1].mean() + 1.5,
            X[y == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
        )
    y = np.choose(y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    plt.show()

# wykres 2D 

def wykres_2d(X, y):
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)


    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(
            X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of IRIS dataset")
    plt.show()

wykres_2d(X, y)
wykres_3d(X, y)