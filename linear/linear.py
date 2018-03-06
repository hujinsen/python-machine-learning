# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
import os


os.chdir(os.path.realpath('.'))

data = loadtxt('ex1data1.txt', delimiter=',')

X = data[:,0] #(97,)
y = data[:,1]


plt.scatter(X, y, marker='x')
plt.show()

m = len(y)

X = np.reshape(X, (m,1)) #现在m为(97, 1)
y = np.reshape(y, (m,1))

X = np.hstack((np.ones((m,1)), X))

theta = np.zeros((2,1))

def computeCost(X: np.ndarray, y: np.ndarray , theta: np.ndarray):
    m = len(y)
    J = 0
    h = np.dot(X, theta)
    J = 1/(2*m) * sum((h-y)**2)
    return J[0]

computeCost(X, y, theta)
computeCost(X, y, np.array([[-1],[2]]))

iterations = 1500;
alpha = 0.01;

def gradientDescent(X, y, theta, alpha, iterations):
     m = len(y)
     J_history = np.zeros((iterations, 1))

     for i in range(iterations):
        h = np.dot(X, theta)
        k = np.dot(np.transpose(X) ,(h-y))
        theta = theta - alpha * k / m
        J_history[i] = computeCost(X, y, theta)
        print(J_history[i])

     return theta, J_history


theta, J_history= gradientDescent(X, y, theta, alpha, iterations);

def plotJ(J_history, iterations):
    x = np.arange(1, iterations+1)
    plt.plot(x, J_history)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('iterations vs loss')
    plt.show()

plotJ(J_history, iterations)

def plot_result(X, y, theta):
    plt.scatter(X[:, 1], y)
    plt.hold(True)
    plt.plot(X[:,1], np.asarray(np.dot(X, theta)), color='r')
    plt.show()

plot_result(X, y, theta)