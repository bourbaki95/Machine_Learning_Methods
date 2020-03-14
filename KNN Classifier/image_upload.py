import os 
import cv2 

data = []
labels = []
dataset_path = "animals"
class_folders = os.listdir(dataset_path)

for class_name in class_folders:
    image_list = os.listdir(dataset_path +  '/' + class_name)
    for image in image_list:
        image = cv2.imread(dataset_path +  '/' + class_name + '/' + image)
        image = cv2.resize(image, (32, 32),interpolation=cv2.INTER_CUBIC)
        data.append(image)
        labels.append(class_name)
print(len(data))
print(len(labels))
print( labels [-1])