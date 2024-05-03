# OPENCV I DOMINUJĄCY KOLOR ZIELONY

import cv2
import matplotlib.pyplot as plt

# Funkcja do obliczania wartości szarego piksela na podstawie pierwszej intuicji
def gray_average(img):
    return (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3.0

# Funkcja do obliczania wartości szarego piksela na podstawie lepszego wzoru
def gray_weighted(img):
    return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

img = cv2.imread('marek.jpg')

gray_avg = gray_average(img)
gray_wtd = gray_weighted(img)

plt.figure(figsize=(10,5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Oryginalny obraz')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray_avg, cmap='gray')
plt.title('Pierwsza intuicja')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gray_wtd, cmap='gray')
plt.title('Lepszy wzór')
plt.axis('off')

plt.show()