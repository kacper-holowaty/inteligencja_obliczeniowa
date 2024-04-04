# DRZEWA DECYZYJNE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris1.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7,random_state=1)

train_inputs = train_set[:, 0:4] 
train_classes = train_set[:, 4] 
test_inputs = test_set[:, 0:4] 
test_classes = test_set[:, 4]

dtc = DecisionTreeClassifier()
dtc = dtc.fit(train_inputs, train_classes)
print(dtc.score(test_inputs, test_classes)*100,'%')

plt.figure(figsize=(15, 10))
plot_tree(dtc, filled=True, feature_names=df.columns[:-1], class_names=df['variety'].unique())
plt.show()


predictions = dtc.predict(test_inputs)
cm = confusion_matrix(test_classes, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=df['variety'].unique(), yticklabels=df['variety'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()