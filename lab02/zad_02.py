# PCA

# from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.decomposition import PCA 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# df = pd.read_csv("iris1.csv")
# column_names = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
# iris_values = df.loc[:, column_names].values
# variety = df.loc[:,['variety']].values
# mapa = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
# y = df['variety'].map(mapa).values
# print(target)
iris = datasets.load_iris() 
X = pd.DataFrame(iris.data, columns=iris.feature_names) 
y = pd.Series(iris.target, name='FlowerType') 
# print(iris.data)
# print(X.head()) 

# pca_iris = PCA(n_components=3).fit(iris_values) 
# print(pca_iris) 
# print(pca_iris.explained_variance_ratio_) 
# print(pca_iris.components_) 
# print(pca_iris.transform(iris_values)) 

# wykres 3D


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
    # Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

plt.show()

# wykres_3d()
# wykres 2D 
def wykres_2d():
    # nie działa (nie wyświetla kropek na wykresie)

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(X)

    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, df[['variety']]], axis = 1)
    

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets = ['setosa', 'versicolor', 'virginica']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['variety'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()