#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:03:19 2018

@author: JSen
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
import os
from scipy.optimize import minimize
from sklearn import linear_model

os.chdir('/Users/JSen/Documents/logistic/')

data = loadtxt('ex2data1.txt', delimiter=',')

X = data[:, :-1]
#最后一列为label
y = data[:, -1:]
y = np.ravel(y)

logistic = linear_model.LogisticRegression()
logistic.fit(X, y)

print(logistic.coef_)
