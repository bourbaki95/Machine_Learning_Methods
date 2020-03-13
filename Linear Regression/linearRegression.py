"""
Andres Restrepo
Linear Regression Model trained by Batch Gradient Descent
"""


import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(10,1)
y_data = 2*x_data+50+5*np.random.random()


b_vector = np.arange(0,100,1) 
w_vector = np.arange(-5, 5,0.1) 
Z = np.zeros((len(b_vector),len(w_vector)))
 
for i in range(len(b_vector)):
    for j in range(len(w_vector)):
        b = b_vector[i]
        w = w_vector[j]
        Z[j][i] = 0        
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (w*x_data[n]+b - y_data[n])**2 # this is the loss 
        Z[j][i] = Z[j][i]/len(x_data)


plt.xlim(0,100)
plt.ylim(-5,5)
plt.contourf(b_vector, w_vector, Z, 50, alpha =0.5, cmap = plt.get_cmap('jet'))
plt.show()


learning_rate = 0.0003
b = 0
b_history = [b] 
w = 1
w_history = [w]
iterations = 15000

def loss_function(w,b):
    loss = 0
    for i in range(len(x_data)):
        loss += (w*x_data[i]+b - y_data[i])**2
    return 0.5*loss

current_loss = loss_function(w,b)
targeted_loss = 0.05

j = 0

while j < iterations and current_loss > targeted_loss :
    dw = 0
    db = 0
    for i in range(len(x_data)):       
        dw += x_data[i]*(w*x_data[i]+b-y_data[i])
        db += w*x_data[i]+b-y_data[i]
    w = w - learning_rate*dw
    w_history.append(w)
    b = b - learning_rate*db
    b_history.append(b)
    current_loss = loss_function(w,b)
    j += 1
    print("w={}, b={}, iteration= {}, loss={}".format(w,b,j, current_loss))   

print(x_data)
print(y_data)
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5,color='black')
plt.show()

