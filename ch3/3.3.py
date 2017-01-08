#coding: utf-8
import numpy as np
import math

def predict(beta, x):
    result = sigmoid(np.linalg.det(x * beta.T))
    if(result > 0.5):
        return 1
    return 0

def sigmoid(x):
    y = float(1) / float(1 + math.e ** (-x))
    return y

def newton_method(beta, x, y):
    col = beta.shape[1]
    der_beta = np.mat(np.zeros((1, col)))
    der_beta_2 = 0
    i = 0
    while(i < len(x)):
        tmp = x[i] * beta.T
        tmp_num = np.linalg.det(tmp)

        p1 = sigmoid(tmp_num)
        der_beta += -x[i] * (np.linalg.det(y[i]) - p1)

        der_beta_2 += x[i] * x[i].T * p1 * (1 - p1)
        i = i + 1

    beta = beta - (1.0 / der_beta_2) * der_beta
    return beta

def cal_best_w_liner(x, y):
    try:
        x_m = x.T * x
        x_mi = x_m * x_m.I
        w = x_mi * x.T * y
        print w
    except:
        print "Error"

    return w

if __name__ == "__main__":
    a1 = np.array([
        [0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.666, 0.091, 1],
        [0.243, 0.267, 1],
        [0.245, 0.057, 1],
        [0.343, 0.099, 1],
        [0.639, 0.161, 1],
        [0.657, 0.198, 1]
        ])
    a2 = np.array([
        [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0]
        ])
    x = np.mat(a1)
    y = np.mat(a2)
    beta = np.mat(np.random.rand(1, 3))
    #print a2
    #cal_best_w_liner(x, y)
    i = 10000
    while(i > 0):
        beta = newton_method(beta, x , y)
        i = i - 1

    a3 = np.array([
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.360, 0.370, 1],
        [0.593, 0.042, 1],
        [0.719, 0.103, 1]
        ])
    x_test = np.mat(a3)
    for x in x_test:
        print predict(beta, x)
    #y = sigmod(0)
    #print y
