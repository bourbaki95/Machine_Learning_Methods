"""
Andres Restrepo
Logistic Regression Model
"""


import numpy as np

# X = np.random.rand(10,3)
X = [ [1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15], [16,17,18], [19,20,21]]
X = np.array(X)


def function(X):
    
    y = []
    for i in range(len(X)):
        y_value = X[i][0] + X[i][1] + X[i][2]
        y.append(y_value)
    return np.array(y)

Y = function(X)

def sigmoid_function(X):

    y = []
    for i in range(len(X)):
        y_value = 1/(1 + np.exp(-X[i]))
        y.append(y_value)
    return np.array(y)

def accuracy(true_positive, true_negative):

    accurate = (true_positive + true_negative)/len(X)
    return accurate

  
print(X)
print(Y)
print(sigmoid_function(Y))

