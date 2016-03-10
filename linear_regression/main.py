# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import random

def batch_gradient_descent(X, Y):
    theta = np.zeros((2, 1))
    # 学习设为0.1的时候程序跑飞
    alpha = 0.001
    itera = 1000
    m = X.shape[0]
    J = []
    err = 0

    print("m:{}".format(m))
    for i in range(itera):
        J_diff_0 = 0
        J_diff_1 = 0
        err = 0
        for j in range(m):
            err = err + (X[j,:].dot(theta) - Y[j])**2
            J_diff_0 = J_diff_0 + (X[j, :].dot(theta) - Y[j]) * X[j, 0]
            J_diff_1 = J_diff_1 + (X[j, :].dot(theta) - Y[j]) * X[j, 1]
        theta[0,0] = theta[0,0] - alpha * J_diff_0
        theta[1,0] = theta[1,0] - alpha * J_diff_1
        print(err/2*m)
    plt.figure(1)
    plt.scatter(X[:, 1], Y,  color='black')
    plt.plot(X[:, 1], X.dot(theta))
    plt.show()

def sgd_gradient_descent(X, Y):
    theta = np.zeros((2, 1))
    # 学习设为0.1的时候程序跑飞
    alpha = 0.01
    itera = 2000000
    m = X.shape[0]
    print("m:{}".format(m))
    for i in range(itera):
        j = random.randint(0,421)
        J_diff_0 = 0
        J_diff_1 = 0
        J_diff_0 = J_diff_0 + (X[j, :].dot(theta) - Y[j]) * X[j, 0]
        J_diff_1 = J_diff_1 + (X[j, :].dot(theta) - Y[j]) * X[j, 1]
        theta[0,0] = theta[0,0] - alpha * J_diff_0
        theta[1,0] = theta[1,0] - alpha * J_diff_1

    plt.figure(1)
    plt.scatter(X[:, 1], Y,  color='black')
    plt.plot(X[:, 1], X.dot(theta))
    plt.show()

def sklearn_linear_model(X, Y):
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    plt.figure(2)
    plt.scatter(X[:,1], Y,  color='black')
    plt.plot(X[:,1], X.dot(regr.coef_) + regr.intercept_)
    plt.show()


if __name__ == "__main__":

    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    print(diabetes_X.shape)
    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    add_one = np.ones((422,1))

    diabetes_X_train = np.concatenate((add_one, diabetes_X_train), axis=1)
    print diabetes_X_train
    diabetes_X_test = diabetes_X[-20:]


    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]


    diabetes_y_test = diabetes.target[-20:]

    print(diabetes_y_train.shape)

    batch_gradient_descent(diabetes_X_train, diabetes_y_train)
    sklearn_linear_model(diabetes_X_train, diabetes_y_train)
    sgd_gradient_descent(diabetes_X_train, diabetes_y_train)

