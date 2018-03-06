#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:28:50 2018

@author: JSen
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
import os


os.chdir(os.path.realpath('.'))

data = loadtxt('ex1data2.txt', delimiter=',')

X = data[:, :-1]
#最后一列为label
y = data[:, -1:]

def feature_normalization(X):
    X_norm = X

    column_mean = np.mean(X_norm, axis=0)
    print('mean=', column_mean)
    column_std = np.std(X_norm, axis=0)
    print('std=',column_std)

    X_norm = X_norm - column_mean
    X_norm = X_norm / column_std

    return column_mean, column_std, X_norm

means, stds, X_norm = feature_normalization(X)

def plot_after_feature_normalization(X):
    plt.scatter(X[:,0], X[:,1])
    plt.show()

plot_after_feature_normalization(X_norm)

m = len(y)
X_norm = np.hstack((np.ones((m,1)), X_norm))

theta = np.zeros((X_norm.shape[1], 1))

def computeCost(X: np.ndarray, y: np.ndarray , theta: np.ndarray):
    m = len(y)
    J = 0
    h = np.dot(X, theta)
    J = 1/(2*m) * sum((h-y)**2)
    return J[0]

computeCost(X_norm, y, theta)

alpha = 0.01;
iterations = 400;

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


theta, J_history= gradientDescent(X_norm, y, theta, alpha, iterations)

def plotJ(J_history, iterations):
    x = np.arange(1, iterations+1)
    plt.plot(x, J_history)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('iterations vs loss')
    plt.show()

plotJ(J_history, iterations)


def test(means, stds, theta):

    t1 = np.array([[1650,3]])

    t1 = t1 - means
    t1 = t1 / stds
    t1 = np.hstack((np.ones((t1.shape[0], 1)), t1))

    res = np.dot(t1, theta)
    print('predict house price:', res[0][0])

test(means, stds, theta)
