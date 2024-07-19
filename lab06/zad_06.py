# DETEKCJA OBIEKTÓW 

# Algorytm YOLO (You Only Look Once):
# Algorytm dzieli obraz na siatkę komórek i dla każdej komórki przewiduje obecność i położenie 
# obiektów poprzez regresję prostokątnych ram, jednocześnie przewidując prawdopodobieństwo klasy obiektu.
# W przeciwieństwie do innych metod, YOLO dokonuje detekcji obiektów w jednym przebiegu sieci neuronowej, 
# co prowadzi do znacznie szybszej detekcji.

from imageai.Detection import ObjectDetection  

recognizer = ObjectDetection()  
path_model = "./models/tiny-yolov3.pt"  
path_input = "./input/image.jpg"  
path_output = "./output/detected_objects.jpg"  

recognizer.setModelTypeAsTinyYOLOv3()  
recognizer.setModelPath(path_model)  
recognizer.loadModel()  

recognition = recognizer.detectObjectsFromImage(  
    input_image = path_input,  
    output_image_path = path_output  
    )  

for item in recognition:  
    print(item["name"] , " : ", item["percentage_probability"])  

