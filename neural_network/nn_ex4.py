#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:47:40 2018

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
theta = spio.loadmat('ex4weights.mat')

theta1 = theta['Theta1'] #25x401
theta2 = theta['Theta2'] #10x26

#load training data
data = spio.loadmat('ex4data1.mat')

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

def sigmoid(z):
    s = 1.0/(1.0 + np.exp(-z))
    return s

input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;

def compress_theta_in_one_column(theta1, theta2):
    t1 = theta1.reshape(-1, 1)
    t2 = theta2.reshape(-1, 1)
    t3 = np.vstack((t1, t2))
    return t3

def decompress_theta_from_cloumn(one_column_theta, input_layer_size, hidden_layer_size, num_labels):
    one_column_theta = one_column_theta.reshape(-1, 1) #确保是nx1向量

    t1 = one_column_theta[0:hidden_layer_size*(input_layer_size+1), :]
    t1 = t1.reshape((hidden_layer_size, input_layer_size+1))
    t2 = one_column_theta[hidden_layer_size*(input_layer_size+1):, :]
    t2 = t2.reshape(((num_labels, hidden_layer_size+1)))
    return t1, t2

def decompress_theta_from_row(one_row_theta, input_layer_size, hidden_layer_size, num_labels):
    one_row_theta = one_row_theta.reshape(1, -1) #确保是1xn向量

    t1 = one_row_theta[:, 0:hidden_layer_size*(input_layer_size+1)]
    t1 = t1.reshape((hidden_layer_size, input_layer_size+1))
    t2 = one_row_theta[:, hidden_layer_size*(input_layer_size+1):]
    t2 = t2.reshape(((num_labels, hidden_layer_size+1)))
    return t1, t2

def nnCostFunction(nn_parms, input_layer_size, hidden_layer_size, num_labels, X, y):
    '''不带正则化的损失函数'''
    theta_t1, theta_t2 = decompress_theta_from_cloumn(nn_parms, input_layer_size, hidden_layer_size, num_labels)

    m = X.shape[0]
    X = np.hstack((np.ones((m,1)), X))
    h1 = np.dot(X, theta_t1.T) #隐层 5000x25
    h1 = sigmoid(h1) #隐层输出
    #为隐层结点添加偏置1
    ones = np.ones((m, 1))
    h1 =  np.hstack((ones, h1)) #5000x26
    h2 = np.dot(h1, theta_t2.T) #5000x10
    h = sigmoid(h2) #结果映射到0-1，之间，后面才可以计算损失

    #将y转化为5000x10矩阵
    y_mat = np.zeros((m, num_labels),dtype=int)
    for i in range(m):
        y_mat[i, y[i]-1] = 1 #1->y(1)=1, 2->y(2)=1,...9->y(9)=1,10->y(10)=1,但python数组从0开始，故每个减一，注意10代表0

    #计算损失
    A = np.dot(y_mat.T, np.log(h)) + np.dot((1-y_mat).T , np.log(1-h)) #10 by 10 matrix
    J = -1/m * np.trace(A)

    return J


theta_in_one_col = compress_theta_in_one_column(theta1, theta2)
loss = nnCostFunction(theta_in_one_col, input_layer_size, hidden_layer_size, num_labels, X, y)
print('loss:', loss)

def nnCostFunction_with_regularization(nn_parms, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coe):
    '''带正则化的损失函数'''
    theta_t1, theta_t2 = decompress_theta_from_cloumn(nn_parms, input_layer_size, hidden_layer_size, num_labels)

    m = X.shape[0]
    X = np.hstack((np.ones((m,1)), X))
    h1 = np.dot(X, theta_t1.T) #隐层 5000x25
    h1 = sigmoid(h1) #隐层输出
    #为隐层结点添加偏置1
    ones = np.ones((m, 1))
    h1 =  np.hstack((ones, h1)) #5000x26
    h2 = np.dot(h1, theta_t2.T) #5000x10
    h = sigmoid(h2) #结果映射到0-1，之间，后面才可以计算损失

    #将y转化为5000x10矩阵
    y_mat = np.zeros((m, num_labels),dtype=int)
    for i in range(m):
        y_mat[i, y[i]-1] = 1 #1->y(1)=1, 2->y(2)=1,...9->y(9)=1,10->y(10)=1,但python数组从0开始，故每个减一，注意10代表0

    #计算损失
    A = np.dot(y_mat.T, np.log(h)) + np.dot((1-y_mat).T , np.log(1-h)) #10 by 10 matrix
    J = -1/m * np.trace(A)
    #正则化,所有theta第一列不用正则化
    theta_t1_r1 = theta_t1[:, 1:]
    theta_t2_r2 = theta_t2[:, 1:]
    B = np.dot(theta_t1_r1, theta_t1_r1.T) + np.dot(theta_t2_r2.T, theta_t2_r2)#25x25
    reg =lambda_coe/(2*m) * np.trace(B)

    J = J + reg

    return J

loss = nnCostFunction_with_regularization(theta_in_one_col, input_layer_size, hidden_layer_size, num_labels, X, y, 1)
print('loss with regularization:', loss)

def sigmoid_gradient(z):
    '''假设z是经过sigmoid函数处理过得'''
#    gz = sigmoid(z)
    g = z * (1-z)
    return g

def randInitializeWeights(input_layer_size, hidden_layer_size):
    epsilon_init = 0.12
    rand_matrix = np.random.rand(hidden_layer_size, input_layer_size+1) #得到[0,1)之间均匀分布的数字
    W = rand_matrix * 2 * epsilon_init - epsilon_init #得到(-epsilon_init, epsilon_init)之间的数字
    return W

def compress_theta_in_one_row(theta1, theta2):
    t1 = np.matrix.flatten(theta1)
    t2 = np.matrix.flatten(theta2)
    t3 = np.hstack((t1, t2)).reshape(1, -1) #连接起来
    return t3


def compute_gradient(nn_parms: np.ndarray, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coe=0):
    '''计算参数为nn_parms，梯度'''
    theta_t1, theta_t2 = decompress_theta_from_cloumn(nn_parms, input_layer_size, hidden_layer_size, num_labels)

    m = X.shape[0]
    X = np.hstack((np.ones((m,1)), X))
    h1 = np.dot(X, theta_t1.T) #隐层 5000x25
    h1 = sigmoid(h1) #隐层输出
    #为隐层结点添加偏置1
    ones = np.ones((m, 1))
    h1 =  np.hstack((ones, h1)) #5000x26
    h2 = np.dot(h1, theta_t2.T) #5000x10
    h = sigmoid(h2) #结果映射到0-1，之间，后面才可以计算损失

    #将y转化为5000x10矩阵
    y_mat = np.zeros((m, num_labels),dtype=int)
    for i in range(m):
        y_mat[i, y[i]-1] = 1 #1->y(1)=1, 2->y(2)=1,...9->y(9)=1,10->y(10)=1,但python数组从0开始，故每个减一，注意10代表0
    '''BP算法'''
    big_delta1 = np.zeros((theta_t1.shape)) #25x401
    big_delta2 = np.zeros((theta_t2.shape)) #10x26
    for i in range(m):
        out = h[i, :]
        y_current = y_mat[i, :]

        delta3 = (out - y_current).reshape(1, -1) #1x10

        delta2 = np.dot(delta3, theta_t2) * sigmoid_gradient(h1[i, :]) #1x26
        delta2 = delta2.reshape(1, -1)

        big_delta2 = big_delta2 + np.dot(delta3.T, h1[i, :].reshape(1, -1)) #10x26

        #此处tricky，分开写
        t1 = delta2[:, 1:].T
        t2 = X[i,:].reshape(1,-1)
        big_delta1 = big_delta1 + np.dot(t1, t2) #25x401

    B1 = np.hstack((np.zeros((theta_t1.shape[0], 1)), theta_t1[:, 1:])) #25x401
    theta1_grad = 1/m * big_delta1 + lambda_coe/m * B1

    B2 = np.hstack((np.zeros((theta_t2.shape[0], 1)), theta_t2[:, 1:])) #10x26
    theta2_grad = 1/m * big_delta2 + lambda_coe/m * B2
    #返回展平的参数
    r = compress_theta_in_one_column(theta1_grad, theta2_grad)

    return r.ravel() #将返回值展成(n,)形式，不能写成(n,1)，因为报维度出错:deltak = numpy.dot(gfk, gfk)

def debugInitializeWeights(fan_out, fan_in):
    '''随机创建一组根据fan_out, fan_in确定的权重'''
    arr = np.arange(1, fan_out*(fan_in + 1) + 1)
    W = np.sin(arr).reshape(fan_out, fan_in + 1)
    return W
test_pram = debugInitializeWeights(3,3)

def computeNumericalGradient(nn_param, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coe):
    '''由导数定义计算对每一个theta的偏导数'''
    numgrad = np.zeros((nn_param.shape))
    perturb = np.zeros((nn_param.shape))

    e = 1e-4
    for p in range(np.size(nn_param)):
        perturb[p, :] = e
        loss1 = nnCostFunction_with_regularization(nn_param - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coe)
        loss2 = nnCostFunction_with_regularization(nn_param + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coe)

        numgrad[p, :] = (loss2 - loss1) / (2 * e)
        perturb[p, :] = 0

    return numgrad

def checkNNGradients(lambda_coe = 0):
    '''创建一个小型网络，验证梯度计算是正确的'''
    input_layer_size = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m = 5;

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1);
    #生成对应的label
    y  = 1 + np.mod(np.arange(1, m+1), num_labels).reshape(-1,1)

    param = compress_theta_in_one_column(Theta1, Theta2)

    cost = nnCostFunction_with_regularization(param, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coe)
    grad = compute_gradient(param, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coe)

    numgrad = computeNumericalGradient(param, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coe)
    #相对误差小于1e-9 特别注意：numgrad和grad相减，一定要确保他们的shape是一样的
    diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)
    print(f'Relative Difference:{diff}')
checkNNGradients()



lambda_coe = 0.6

initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_theta_in_one_col = compress_theta_in_one_column(initial_theta1, initial_theta2)

t1, t2 = decompress_theta_from_cloumn(initial_theta_in_one_col, input_layer_size, hidden_layer_size, num_labels)


nnCostFunction_with_regularization(initial_theta_in_one_col, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coe)
g = compute_gradient(initial_theta_in_one_col, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coe)

'''最小化损失'''
result = optimize.fmin_cg(nnCostFunction_with_regularization, initial_theta_in_one_col, fprime=compute_gradient, args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambda_coe), maxiter=100)
#最终结果
res_theta1, res_theta2 = decompress_theta_from_cloumn(result, input_layer_size, hidden_layer_size, num_labels)

def predict(theta1, theta2, X):
    m = X.shape[0]

    #forward propagation
    X = np.hstack((np.ones((m,1)), X))
    h1 = np.dot(X, theta1.T) #隐层 5000x25
    h1 = sigmoid(h1) #隐层输出
    #为隐层结点添加偏置1
    ones = np.ones((m, 1))
    h1 =  np.hstack((ones, h1)) #5000x26
    h2 = np.dot(h1, theta2.T) #5000x10
    h = sigmoid(h2) #结果映射到0-1，之间，后面才可以计算损失

    #找出每行最大值得下标，因为下标从0开始，而预测结果从1开始，故加一，
    row_max_index = np.argmax(h, axis=1)
    row_max_index = row_max_index + 1
    return row_max_index

pre = predict(res_theta1, res_theta2, X)
print('accuracy:', np.mean(np.float32(pre.reshape(-1,1)==y)))