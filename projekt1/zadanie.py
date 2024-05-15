# ROZPOZNANIE ILE ZARABIA OSOBA - ZADANIE PROJEKTOWE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Preprocessing
df = pd.read_csv("adult.csv")

# print(df)
# print(df.describe())

df.replace("?", pd.NA, inplace=True)
print(df.isnull().sum())
for column in df.columns:
    if df[column].dtype == 'object':  
        most_common_value = df[column].mode()[0]
        df[column].fillna(most_common_value, inplace=True)

# print(df.duplicated().sum())
# print(df.drop_duplicates(inplace=True))
# print(df.isnull().sum())
# print(df.duplicated().sum())

# print(df.head())

df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
df['marital.status'] = df['marital.status'].replace(['Never-married', 'Divorced', 'Separated', 'Widowed'], 'Single')
df['marital.status'] = df['marital.status'].replace(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')
df['marital.status'] = df['marital.status'].map({'Married': 1, 'Single': 0})

df['occupation'] = df['occupation'].replace(['Prof-specialty', 'Exec-managerial', 'Tech-support'], 'White-collar')
df['occupation'] = df['occupation'].replace(['Craft-repair', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing'], 'Blue-collar')
df['occupation'] = df['occupation'].replace(['Adm-clerical', 'Sales', 'Other-service', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'], 'Service')
df['occupation'] = df['occupation'].replace(['White-collar','Blue-collar','Service'],[0,1,2])

df['relationship'] = df['relationship'].replace(['Husband', 'Wife', 'Own-child'], 'Family')
df['relationship'] = df['relationship'].replace(['Not-in-family', 'Unmarried', 'Other-relative'], 'Non-Family')
df['relationship'] = df['relationship'].map({'Family': 1, 'Non-Family': 0})

df['race'] = df['race'].replace(['White'], 'White')
df['race'] = df['race'].replace(['Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], 'Non-White')
df['race'] = df['race'].map({'White': 1, 'Non-White': 0})

df.loc[df['native.country'] != 'United-States', 'native.country'] = 'Other'
df['native.country'] = df['native.country'].map({'United-States': 1, 'Other': 0})

df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

columns = ['age', 'education.num', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'hours.per.week', 'native.country', 'income']
df = df[columns]

# for column in df.columns:
#     print(df[column].value_counts())

# print(df)

# print("\n==========================================================================================================\n")

# Klasyfikacja

X = df.drop(['income'], axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Decision Tree Classifier
def dtc_classification():
    dtc = DecisionTreeClassifier(random_state=0)
    dtc = dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Klasyfikacja metodą drzewa decyzyjnego:")
    print("Dokładność:", accuracy)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=df['income'].unique(), yticklabels=df['income'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix Decision Tree')
    plt.show()

# dtc_classification()

# Klasyfikacja metodą k-najbliższych sąsiadów 
def knn_classification(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"k-NN, k={k}")
    print("Dokładność:", accuracy*100, '%')

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=df['income'].unique(), yticklabels=df['income'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (k={k})')
    plt.show()


# knn_classification(15)

# Klasyfikacja metodą Naive Bayes
def naive_bayes_classification():
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    misclassified = (y_test != y_pred).sum()
    total_points = X_test.shape[0]
    accuracy = ((total_points - misclassified) / total_points) * 100
    print("Dla klasyfikatora Naive Bayes")
    print("Dokładność: %.2f%%" % accuracy)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=df['income'].unique(), yticklabels=df['income'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix Naive Bayes')
    plt.show()

# naive_bayes_classification()

# Sieć neuronowa
def mlp_neural_network_classifier():
    clf = MLPClassifier(hidden_layer_sizes=(3, 2), activation='relu', max_iter=500, random_state=1)

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    print('Dokładność sieci neuronowej:',metrics.accuracy_score(prediction,y_test))
    cm = confusion_matrix(y_test, prediction)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=df['income'].unique(), yticklabels=df['income'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix Neural Network')
    plt.show()

# mlp_neural_network_classifier()

def keras_neural_network():
    y_array = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder()
    y_encoded = encoder.fit_transform(y_array.reshape(-1, 1)).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(y_encoded.shape[1], activation='softmax')
    ])

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', color='grey')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', color='grey')
    plt.legend()

    plt.tight_layout()
    plt.show()

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=df['income'].unique(), yticklabels=df['income'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix Neural Network')
    plt.show()

# keras_neural_network()
