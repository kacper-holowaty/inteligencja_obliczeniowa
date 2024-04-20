# SIEĆ I IRYSY

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

data = pd.read_csv("iris1.csv")

df_norm = data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
target = data[['variety']].replace(['Setosa','Versicolor','Virginica'],[0,1,2])

df = pd.concat([df_norm, target], axis=1)

(train_set, test_set) = train_test_split(df, train_size=0.7,random_state=13)

trainX = train_set[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
trainY = train_set['variety']
testX = test_set[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
testY = test_set['variety']

clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,), random_state=1)

clf2.fit(trainX, trainY)
prediction2 = clf2.predict(testX)
print(prediction2)
print(testY.values)

print('Dokładność sieci z 2 neuronami:',metrics.accuracy_score(prediction2,testY))

clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,), random_state=1)

clf3.fit(trainX, trainY)
prediction3 = clf3.predict(testX)
print(prediction3)
print(testY.values)

print('Dokładność sieci z 3 neuronami:',metrics.accuracy_score(prediction3,testY))


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,3), random_state=1)

clf.fit(trainX, trainY)
prediction = clf.predict(testX)
print(prediction)
print(testY.values)

print('Dokładność sieci z dwiema warstwami neuronowymi, po 3 neurony każda:',metrics.accuracy_score(prediction,testY))
 
