# INNE KLASYFIKATORY

import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
sns.set()

df = pd.read_csv("iris1.csv")

X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = df['variety']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

k_values = [3, 5, 11]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"k-NN, k={k}")
    print("Dokładność:", accuracy*100, '%')

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=df['variety'].unique(), yticklabels=df['variety'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (k={k})')
    plt.show()



gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
misclassified = (y_test != y_pred).sum()
total_points = X_test.shape[0]
accuracy = ((total_points - misclassified) / total_points) * 100
print("Dla klasyfikatora Naive Bayes")
print("Dokładność: %.2f%%" % accuracy)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=df['variety'].unique(), yticklabels=df['variety'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix Naive Bayes')
plt.show()