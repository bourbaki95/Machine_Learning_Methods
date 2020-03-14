import numpy as np
import os
import cv2
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

def test_percentages(training, validation, testing, data_size):
    train_size = data_size*training
    validation_size = data_size*validation
    partition_size = train_size + validation_size
    ratio_1 = testing
    ratio_2 = validation_size/partition_size
    return  (ratio_1, ratio_2)
    

def train_validation_test(x,y, train_ratio, val_ratio, test_ratio):
    second_ratio = (val_ratio)/(train_ratio + val_ratio)
    train_x, test_x, train_y, test_y = train_test_split(x,y, test_size = test_ratio, random_state = 27)
    train_x_val, test_x_val, train_y_val, test_y_val = train_test_split(train_x,train_y, test_size = second_ratio, random_state = 27)
    return train_x, test_x, train_y, test_y, train_x_val, test_x_val, train_y_val, test_y_val


def knn_classifier(train_x, test_x, train_y, test_y, train_x_val, test_x_val, train_y_val, test_y_val, neighbors, metric):
    cats = 0
    dogs = 0
    pandas = 0

    model = KNeighborsClassifier(n_neighbors=neighbors, p=metric)
    model.fit(train_x, train_y)
    for i in range(len(test_x)):
        prediction = model.predict([test_x[i]])
        if prediction == [0]:
            cats  +=1
        if prediction == [1]:
            dogs  +=1
        if prediction == [2]:
            pandas +=1
    print("Cats={}, Dogs={}, Pandas= {}".format(cats,dogs,pandas))

def knn_validation(train_x_val, test_x_val, train_y_val, test_y_val):

    neighbors = [3,5,7]
    metrics = [1,2]
    for neighbors in neighbors:
        for metric in metrics:
            model = KNeighborsClassifier(n_neighbors=neighbors, p=metric)
            model.fit(train_x_val, train_y_val)

x,y = reading_data()
x,y = loading_data(x,y)
x,y = encoding_data(x,y)
train_x, test_x, train_y, test_y, train_x_val, test_x_val, train_y_val, test_y_val = train_validation_test(x,y, 0.7, 0.1, 0.2)
knn_classifier(train_x, test_x, train_y, test_y, train_x_val, test_x_val, train_y_val, test_y_val, 1, 2)

