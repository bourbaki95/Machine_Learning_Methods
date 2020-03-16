import numpy as np
import os
import cv2
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics

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

def train_validation_test(x,y, train_ratio, val_ratio, test_ratio):
    second_ratio = (val_ratio)/(train_ratio + val_ratio)
    train_x, test_x, train_y, test_y = train_test_split(x,y, test_size = test_ratio, random_state = 27)
    train_x, val_x, train_y, val_y= train_test_split(train_x,train_y, test_size = second_ratio, random_state = 27)
    return train_x, test_x, train_y, test_y, val_x, val_y

def metrics_score(y_test, y_pred): 
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred, average= 'weighted'))
    print("Recall:",metrics.recall_score(y_test, y_pred, average= 'weighted'))
    print("F Score:",metrics.f1_score(y_test,y_pred, average = 'weighted'))

def knn(train_x, test_x, train_y, test_y, neighbors, metrics):
    for neighbor in neighbors:
        for metric in metrics:
            model = KNeighborsClassifier(n_neighbors=neighbor, p=metric)
            model.fit(train_x, train_y)
            y_predict = model.predict(test_x)
                      print("Neighborhood = {}, L{}".format(neighbor, metric))
            metrics_score(test_y, y_predict)
            print()

x,y = reading_data()
x,y = loading_data(x,y)
x,y = encoding_data(x,y)
train_x, test_x, train_y, test_y, val_x, val_y = train_validation_test(x,y, 0.7, 0.1, 0.2)

print("KNN Classifier.")
#Cross validation for K = [3,5,7] and L = [L1,L2]
print('Evaluating perfomance over the predictions in the validation set:')
knn(train_x, val_x, train_y, val_y, [3,5,7], [1,2])
#Testing the KNN classifier over the actual testing set
print('Evaluating perfomance over the predictions in the testing set :')
knn(train_x, test_x, train_y, test_y, [7], [1])


