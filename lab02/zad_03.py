# NORMALIZACJA

import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')

stats = X.describe()
print(stats)

# Oryginalne dane
plt.figure(2, figsize=(8, 6))
plt.clf()
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X[y == i]['sepal length (cm)'], X[y == i]['sepal width (cm)'], s=35, label=target_name)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Original Dataset')
plt.legend()
plt.show()

# Normalizacja Min-Max
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_minmax[y == i][:, 0], X_minmax[y == i][:, 1], s=35, label=target_name)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Min-Max Normalised Dataset')
plt.legend()
plt.show()

# Skalowanie Z-Score
scaler_zscore = StandardScaler()
X_zscore = scaler_zscore.fit_transform(X)

for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_zscore[y == i][:, 0], X_zscore[y == i][:, 1], s=35, label=target_name)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Z-Score Scaled Dataset')
plt.legend()
plt.show()
