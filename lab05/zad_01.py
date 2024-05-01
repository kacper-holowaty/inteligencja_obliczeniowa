# IRYSY W KERAS

import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from tensorflow.keras.models import load_model 
 
iris = load_iris() 
X = iris.data 
y = iris.target 

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
 
encoder = OneHotEncoder() 
y_encoded = encoder.fit_transform(y.reshape(-1, 1)).toarray()
 
# Split the dataset into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, 
random_state=42) 
 
# Load the pre-trained model 
model = load_model('iris_model.h5') 
 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # ponowne kompilowanie kodu, bez tego wyrzuca błąd

# Continue training the model for 10 more epochs 
model.fit(X_train, y_train, epochs=10) 
 
# Save the updated model 
model.save('updated_iris_model.h5') 
 
# Evaluate the updated model on the test set 
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0) 
print(f"Test Accuracy: {test_accuracy*100:.2f}%")


# ODPOWIEDZI

# a) 
# StandardScaler jest narzędziem służącym do skalowania cech danych liczbowych. Jego głównym celem jest przekształcenie
# danych w taki sposób, aby miały średnią równą zero i wariancję równą jeden.
# Proces transformacji danych liczbowych za pomocą StandardScaler przebiega w następujący sposób:
# 1. Obliczane są średnie wartości (mean) każdej cechy (kolumny) danych.
# 2. Obliczane są odchylenia standardowe (standard deviation) każdej cechy.
# 3. Każda wartość w danej kolumnie jest odejmowana od średniej tej kolumny.
# 4. Wynik jest dzielony przez odchylenie standardowe tej kolumny. 

# b)
# Kodowanie "one-hot" polega na tym, że każda unikalna wartość w zmiennej kategorycznej jest przekształcana na wektor
# binarny o długości równej liczbie unikalnych wartości w tej zmiennej. Wartość odpowiadająca danej kategorii ma 
# wartość 1, natomiast wszystkie inne wartości są ustawione na 0.
# Dla każdej etykiety tworzony jest wektor binarny, gdzie pozycja odpowiadająca danej klasie jest ustawiona na 1, 
# a pozostałe pozycje są ustawione na 0.
# Na przykład, dla trzech klas etykiet klas 0, 1 i 2:
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

# c)
# Analizując przedstawiony graficznie model:
# - warstwa wejściowa ma 4 neurony
# - X_train.shape[1] oznacza liczbę cech (kolumn) w danych treningowych
# - warstwa wyjściowa ma 3 neurony
# - w przypadku y_encoded.shape[1], y_encoded to zdekodowane etykiety klas, a y_encoded.shape[1] oznacza liczbę klas

# d)
# Czy funkcja aktywacji 'relu' jest najlepsza do tego zadania? Trudno powiedzieć. 
# Dawała lepsze wyniki (Accuracy, najczęściej 100%) niż 'tanh' oraz 'sigmoid' dla tego konkretnego modelu.

# e)
# Gdy użyłem optymalizatora 'sgd', to wynik (Test Accuracy) wyszedł mi około 86%, gdzie dla optymalizatora 'adam' najczęściej było to 100%. 
# Nie zauważyłem, aby funkcja straty wpływała na dokładność modelu. Zarówno dla 'mean_squared_error' jak i 'categorical_crossentropy' 
# dokładność zazwyczaj jest 100%.
# Tak, możemy dostosować szybkość uczenia się optymalizatora, parametr ten jest zazwyczaj nazywany "learning rate".

# f)
# Możemy zmienić rozmiar partii za pomocą np. batch_size=4 w model.fit(X_train, y_train, batch_size=4, epochs=100, validation_split=0.2)

# g)
# Na podstawie przedstawionych w pdfie wykresów, mogę stwierdzić, że sieć osiągnęła najlepszą wydajność w epoce 40-50.
# Model jest raczej dobrze dopasowany, nie mamy do czynienia z niedouczeniem lub przeuczeniem.

# h)
# Podany kod między innymi przetwarza dane za pomocą StandardScaler i OneHotEncoder, w celu przygotowania ich do
# trenowania sieci neuronowej. Wykonywany jest podział na zbiór treningowy i testowy oraz następnie ładowanie 
# zapisanego wcześniej modelu 'iris_model.h5'. Wczytany model jest następnie trenowany za pomocą funkcji fit.

