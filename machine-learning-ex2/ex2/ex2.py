#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
path = 'E:\\Python\\Andrew NG\\machine-learning-ex2\\ex2\\ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

#
##  ploting data
#pos = data.loc[data['Admitted'] == 1]
#neg = data.loc[data['Admitted'] == 0]
#
#fig, ax = plt.subplots(figsize=(5,5))
#ax.scatter(pos['Exam 1'], pos['Exam 2'], s=50, c='b', marker='o', label='Admitted')
#ax.scatter(neg['Exam 1'], neg['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
#ax.legend()
#ax.set_xlabel('Exam 1 Score')
#ax.set_ylabel('Exam 2 Score')



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    m = len(y)
    g = sigmoid(np.dot(X,theta))
    J = (np.dot(y.T, np.log(g)) + np.dot((1-y).T, np.log(1-g))) / -m    
    return J





def Gradient(theta,x,y):
    m , n = x.shape
    theta = theta.reshape((n,1));
#    y = y.reshape((m,1))
    sigmoid_x_theta = sigmoid(x.dot(theta));
    grad = ((x.T).dot(sigmoid_x_theta-y))/m;
    return grad.flatten();


# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros((X.shape[1],1))


thiscost= cost(theta, X, y)    
grad = Gradient(theta, X, y)

import scipy.optimize as opt
#result = opt.fmin_bfgs(cost, x0=theta, fprime=grad, args=(X, y))
res = opt.minimize(fun = cost, x0 = theta, args = (X, y),method = 'TNC',jac = Gradient);
    

def predict(theta, X):
    p_1 = sigmoid(np.dot(X, theta))
    return p_1 > 0.5

opt_theta = res.x


predictions = predict(opt_theta, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))






