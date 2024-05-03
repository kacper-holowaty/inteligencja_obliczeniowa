# OPENCV I LICZNIK PTAKÓW 

import os
import cv2

folder_path = "bird_miniatures"
images = os.listdir(folder_path)

def count_birds(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

for image_name in images:
    image_path = os.path.join(folder_path, image_name)
    image = cv2.imread(image_path)
    bird_count = count_birds(image)
    print(f"{image_name}: {bird_count}")