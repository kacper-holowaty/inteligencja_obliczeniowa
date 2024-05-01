# SIEĆ I CUKRZYCA

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('diabetes1.csv')

df_norm = data.drop('class', axis=1).apply(lambda x: (x - x.min()) / (x.max() - x.min()))
target = data[['class']].replace(['tested_negative', 'tested_positive'], [0,1])
df = pd.concat([df_norm, target], axis=1)

(train_set, test_set) = train_test_split(df, train_size=0.7, random_state=1)

trainX = train_set.drop('class', axis=1)
trainY = train_set['class']
testX = test_set.drop('class', axis=1)
testY = test_set['class']

clf = MLPClassifier(hidden_layer_sizes=(6, 3), activation='relu', max_iter=500, random_state=1)
clf.fit(trainX, trainY)
prediction = clf.predict(testX)
print(prediction)
print(testY.values)

print('Dokładność: ', accuracy_score(prediction,testY))
cm = confusion_matrix(testY, prediction)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()

# 1. True Positive (TP): Liczba przypadków, w których rzeczywista klasa to pozytywna, a model poprawnie przewidział tę klasę jako pozytywną.
# 2. False Positive (FP): Liczba przypadków, w których rzeczywista klasa to negatywna, a model błędnie przewidział tę klasę jako pozytywną (fałszywy alarm).
# 3. True Negative (TN): Liczba przypadków, w których rzeczywista klasa to negatywna, a model poprawnie przewidział tę klasę jako negatywną.
# 4. False Negative (FN): Liczba przypadków, w których rzeczywista klasa to pozytywna, a model błędnie przewidział tę klasę jako negatywną.

# W tym przypadku: 131 - TN, 36 - FP, 15 - FN, 49 - TP 
# Gorsze są błędy FN, ponieważ błędnie stwierdzamy, że dana osoba nie ma cukrzycy, co może być niebezpieczne jeśli tak naprawdę ją ma.