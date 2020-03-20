import numpy as np
import pandas as pd
import time
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def reading_data_MNIST():   
    data = np.array(pd.read_csv('MNIST.csv'))
    data = np.transpose(data)
    x = data[1:]
    x = x.transpose()
    y = data[0]
    return x,y

def reading_data_BankNote():
    data = np.array(pd.read_csv('Bank_Authentication.csv'))
    data = np.transpose(data)
    x = data[0:4]
    x = x.transpose()
    y = data[4]
    return x,y

def scaling_feature_data(x):
    return preprocessing.scale(x)

def silencing_warnings():
    warnings.filterwarnings('ignore')

def timing(function, *arguments):
    minutes = 0
    seconds = 0
    start = time.time()
    function(*arguments)
    end = time.time()
    elapsed_time = math.ceil(end-start)
    if elapsed_time > 60:
        minutes = elapsed_time//60
        seconds = elapsed_time - 60*minutes
    else:
        seconds = elapsed_time
    print("Total Time Elapsed, Minutes = {}, Seconds = {}".format(minutes, seconds))

def train_validation_test_split(x,y, train_ratio, val_ratio, test_ratio):
    second_ratio = (val_ratio)/(train_ratio + val_ratio)
    train_x, test_x, train_y, test_y = train_test_split(x,y, test_size = test_ratio, random_state = 27)
    train_x, val_x, train_y, val_y= train_test_split(train_x,train_y, test_size = second_ratio, random_state = 27)
    return train_x, test_x, train_y, test_y, val_x, val_y

def metrics_score(test_y, y_predict):
    print(classification_report(test_y, y_predict))   

def knn(train_x, test_x, train_y, test_y, neighbors, metrics):
    for neighbor in neighbors:
        for metric in metrics:
            model = KNeighborsClassifier(n_neighbors=neighbor, p=metric)
            model.fit(train_x, train_y)
            y_predict = model.predict(test_x)
            print("Neighborhood = {}, L{}".format(neighbor, metric))
            metrics_score(test_y, y_predict)
            print()

def logistic_regression(train_x, test_x, train_y, test_y):
    model = LogisticRegression(solver='sag')
    model.fit(train_x, train_y)
    y_predict = model.predict(test_x)
    metrics_score(test_y, y_predict)

def decision_tree(train_x, test_x, train_y, test_y, criteria):
    for criterion in criteria:
        model = DecisionTreeClassifier(criterion= criterion)
        model.fit(train_x, train_y)
        y_predict = model.predict(test_x)
        print("Criterion = {}".format(criterion))
        metrics_score(test_y, y_predict)
        print()

silencing_warnings()
x,y = reading_data_MNIST()
x = scaling_feature_data(x)
train_x, test_x, train_y, test_y, val_x, val_y = train_validation_test_split(x,y, 0.7, 0.1, 0.2)
print('Logistic Regression:')
timing(logistic_regression, train_x, test_x, train_y, test_y)
print('KNN Classifier:')
timing(knn, train_x, test_x, train_y, test_y, [3], [1])
print('Decision Tree:')
timing(decision_tree, train_x, test_x, train_y, test_y, ['entropy'])
