#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:03:20 2018

@author: JSen
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt, load
import os
from scipy import optimize
from scipy.optimize import minimize
from sklearn import linear_model
import scipy.io as spio
import random

os.chdir(os.path.realpath('.'))

#load weights
theta = spio.loadmat('ex3weights.mat')

theta1 = theta['Theta1'] #25x401
theta2 = theta['Theta2'] #10x26

#load training data
data = spio.loadmat('ex3data1.mat')

#X 5000x400   y 5000x1
X = data['X']
y = data['y']

# 显示100个数字,拿某位仁兄代码过来用
def display_data(imgData):
    sum = 0
    '''
    显示100个数（若是一个一个绘制将会非常慢，可以将要画的数字整理好，放到一个矩阵中，显示这个矩阵即可）
    - 初始化一个二维数组
    - 将每行的数据调整成图像的矩阵，放进二维数组
    - 显示即可
    '''
    pad = 1
    display_array = -np.ones((pad+10*(20+pad), pad+10*(20+pad)))
    for i in range(10):
        for j in range(10):
            display_array[pad+i*(20+pad):pad+i*(20+pad)+20, pad+j*(20+pad):pad+j*(20+pad)+20] = (imgData[sum,:].reshape(20,20,order="F"))    # order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
            sum += 1

    plt.imshow(display_array,cmap='gray')   #显示灰度图像
    plt.axis('off')
    plt.show()

#随机选取100个数字
rand_indices = random.choices(range(X.shape[0]), k=100)
display_data(X[rand_indices,:])     # 显示100个数字

input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;

def predict(theta1, theta2, X):
    m = X.shape[0]
    num_labels = theta2.shape[0]
    p = np.zeros((m, 1))

    X = np.hstack((np.ones((m,1)), X))
    h1 = np.dot(X, theta1.T) #隐层 5000x25
    #为隐层结点添加偏置1
    ones = np.ones((h1.shape[0], 1))
    h1 =  np.hstack((ones, h1)) #5000x26
    out = np.dot(h1, theta2.T) #5000x10 本预测可以不用sigmoid函数，因为是比大小，而sigmoid是单调增加函数，故省了
    row_max_index = np.argmax(out, axis=1)
    row_max_index = row_max_index + 1
    return row_max_index

pre = predict(theta1, theta2, X)
print('accurate:', np.mean(np.float32(pre.reshape(-1,1)==y)))