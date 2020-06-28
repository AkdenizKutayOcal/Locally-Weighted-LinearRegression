'''
Akdeniz Kutay Öçal
'''

# Libraries
import operator
import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
import math

# Data Generation
m = 100
X = np.random.rand(m, 1) * 2
y = np.sin(2 * math.pi * X) + np.random.randn(m, 1)


def sort(X, y):  # sorts given X and Y arrays
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, y), key=sort_axis)
    X, y = zip(*sorted_zip)
    return X, y


def weighted_linear_regression(X, y, iterNo, eta, x, tau):
    # used variables
    weights = np.random.random(m)  # weight array for holding weights of each x value
    theta0 = np.random.random(1)  # theta values for 0 and 1
    theta1 = np.random.random(1)
    sum0 = 0  # sum values of instances
    sum1 = 0

    # weight calculation depends on x query point
    for k in range(m):
        weights[k] = math.exp(-1 * (((X[k] - x) ** 2) / (2 * (tau ** 2))))

    for j in range(iterNo):

        for i in range(m):
            gradient0 = ((theta0 + theta1 * X[i] - y[i]) * weights[i])
            gradient1 = ((theta0 + theta1 * X[i] - y[i]) * X[i] * weights[i])
            sum0 += gradient0
            sum1 += gradient1

        sum0 *= 2.0 / m
        sum1 *= 2.0 / m

        theta0 = theta0 - eta * sum0
        theta1 = theta1 - eta * sum1

    return theta0, theta1


# method calls
y_pred = np.random.rand(m, 1)  # list that holds predicted y values

# calls wlr method for all x instances as a query point
for i in range(m):
    theta0, theta1 = weighted_linear_regression(X, y, 100, 0.4, X[i], 10)
    y_pred[i] = theta0 + theta1 * X[i]

X, y_pred = sort(X, y_pred)

# plot the graph
plt.title('Weighted Linear Regression with tau = 10',size=32)
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.show()
