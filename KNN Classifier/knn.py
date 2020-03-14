import numpy as np
import os
import cv2
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def reading_data():   
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
    return (data, labels)

def loading_data(data, labels):
    x = np.array(data)
    y = np.array(labels)
    return (x,y)

def encoding_data(x,y):
    label_encoder = preprocessing.LabelEncoder()
    x = x.reshape((x.shape[0],3072))
    y = label_encoder.fit_transform(y)
    return (x,y)

def print_data(x,y):
    print(x)
    print(y)

def knn_classifier(x,y):
    cats = 0
    dogs = 0
    pandas = 0

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x,y)
    for i in range(len(x)):
        prediction = model.predict([x[i]])
        if prediction == [0]:
            cats  +=1
        if prediction == [1]:
            dogs  +=1
        if prediction == [2]:
            pandas +=1
    print("Cats={}, Dogs={}, Pandas= {}".format(cats,dogs,pandas))


x,y = reading_data()
x,y = loading_data(x,y)
x,y = encoding_data(x,y)
knn_classifier(x,y)

