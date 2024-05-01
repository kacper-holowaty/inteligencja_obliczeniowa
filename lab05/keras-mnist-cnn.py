import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History, ModelCheckpoint

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1)  # Save original labels for confusion matrix

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define model checkpoint callback
# checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history])
# po dodaniu checkpoint:
# model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history, checkpoint])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()


# ODPOWIEDZI:

# a)
# Funkcja reshape jest używana, aby zmienić kształt danych wejściowych. Pierwsza linijka reshape zmienia 
# kształt obrazów treningowych i testowych z (n, 28, 28) na (n, 28, 28, 1), gdzie n oznacza liczbę obrazów, 
# co oznacza, że dodaje się jeden wymiar do końca, aby wskazać, że obrazy są w skali szarości.
# Przeprowadzana jest też normalizacja, czyli wszystkie piksele obrazów są normalizowane do wartości 
# z zakresu od 0 do 1 przez podzielenie przez 255, co pomaga w szybszym uczeniu się modelu.

# Funkcja to_categorical konwertuje etykiety klas (liczby od 0 do 9) na postać "one-hot encoded", co oznacza, 
# że każda liczba jest zamieniana na wektor zer i jednej jedynki, gdzie jedynka wskazuje na klasę danej liczby. 
# Na przykład liczba 3 zostałaby zakodowana jako [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].

# Funkcja np.argmax jest używana do przekształcenia danych w formie one-hot encoding z powrotem na postać liczbową.
# Używamy np.argmax wraz z odpowiednim parametrem osi, aby uzyskać indeks największej wartości wzdłuż osi klas. 
# Na przykład, jeśli przewidywania modelu to [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], np.argmax zwróci 7, co odpowiada etykiecie klasy 7.

# b)
# W modelu sieci neuronowej mamy:

# 1. Warstwa konwolucyjna (Conv2D):
# Wejście: Obrazy o wymiarach (28, 28, 1), które są mapami cech (mapami pikseli).
# Co się dzieje: Warstwa konwolucyjna przekształca obrazy, stosując zestaw filtrów konwolucyjnych, aby wyodrębnić cechy lokalne. 
# Filtry te przesuwają się po obrazie, obliczając iloczyny punktowe z fragmentami obrazu. Rezultatem jest zestaw tzw. "map cech".
# Wyjście: Zestaw map cech o wymiarach zależnych od liczby i konfiguracji filtrów oraz użytych parametrów, na przykład (26, 26, 32) 
# w przypadku tego modelu (32 oznacza liczbę filtrów).

# 2. Warstwa max pooling (MaxPooling2D):
# Wejście: Mapy cech wygenerowane przez warstwę konwolucyjną.
# Co się dzieje: Warstwa max pooling redukuje wymiary map cech, zachowując jedynie najważniejsze informacje. W przypadku 
# tej warstwy, obszar o wymiarach (2, 2) jest przetwarzany, a największa wartość z obszaru jest zachowywana.
# Wyjście: Mapy cech o zmniejszonych wymiarach, na przykład (13, 13, 32) w przypadku tego modelu.

# 3. Warstwa Flatten:
# Wejście: Mapy cech z warstwy poprzedniej.
# Co się dzieje: Ta warstwa spłaszcza dane z wielowymiarowej postaci do jednowymiarowej, przygotowując je do 
# przekazania do warstw w pełni połączonych (Dense).
# Wyjście: Jednowymiarowy wektor danych.

# 4. Warstwy w pełni połączone (Dense):
# Wejście: Jednowymiarowy wektor danych pochodzący z warstwy Flatten lub z poprzedniej warstwy Dense.
# Co się dzieje: Warstwy Dense wykonują operacje liniowe na danych, po których stosowana jest funkcja aktywacji, 
# w tym przypadku funkcja ReLU w warstwie Dense z 64 neuronami i funkcja softmax w warstwie wyjściowej z 10 neuronami.
# Wyjście: Wektor wynikowy zawierający przewidywane etykiety klas.

# c)
# Najczęściej są mylone ze sobą cyfry 4 i 9 oraz 3 i 5.

# d)
# W tym przypadku mamy do czynienia z przeuczeniem. Krzywa dokładności na zbiorze treningowym stale rośnie, natomiast
# krzywa dokładności na zbiorze walidacyjnym utrzymuje się na stałym poziomie po osiągnięciu pewnego punktu.

