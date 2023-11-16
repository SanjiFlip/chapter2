# -*- coding: UTF-8 -*-
"""
@Project ：chapter2 
@File    ：test_fun_new.py
@Author  ：wangyi
@Date    ：2023/11/8 19:01 
"""
import numpy as np
import math
import random

"""
def fitness(x):
    return x[0] * np.sin(x[0]) * np.cos(x[0] * 2) - 2 * x[0] * np.sin(3 * x[0]) + 3 * x[0] * np.sin(4 * x[0])
"""

# 单峰值


'''
Sphere Function
30
[-100,100]
min = 0
'''

F1_lb = [-100]
F1_ub = [100]


def F1(x):
    o = np.sum(x * x)
    return o


'''
Schwefel's Problem 2.22
30
[-10,10]
min = 0
'''

F2_lb = [-10]
F2_ub = [10]


def F2(x):
    o = np.sum(np.abs(x)) + np.prod(np.abs(x))
    return o


'''
Schwefel's Problem 1.2
30
[-100,100]
min = 0
'''

F3_lb = [-100]
F3_ub = [100]


def F3(x):
    dim = x.shape[0]
    result = 0
    for i in range(dim):
        result = result + np.sum(x[0:i + 1]) ** 2
    return result


'''
Schwefel's Problem 2.21
30
[-10,10]
min = 0
'''

F4_lb = [-10]
F4_ub = [10]


def F4(X):
    O = np.max(np.abs(X))
    return O


'''
Generalized Rosenbrock's Function
[-30,30]
min = 0
'''

F5_lb = [-30]
F5_ub = [30]


def F5(x):
    dim = x.shape[0]
    result = np.sum(100 * (x[1:dim] - (x[0:dim - 1] ** 2)) ** 2 + (x[0:dim - 1] - 1) ** 2)
    return result


'''
Step Function
[-100,100]
min = 0
'''

F6_lb = [-100]
F6_ub = [100]


def F6(x):
    result = np.sum(np.abs(x + 0.5) ** 2)
    return result


'''
Quartic Function i.e. Noise
[-1.28,1.28]
min = 0
'''

F7_lb = [-1.28]
F7_ub = [1.28]


def F7(x):
    dim = x.shape[0]
    Temp = np.arange(1, dim + 1, 1)
    result = np.sum(Temp * (x ** 4)) + np.random.random()
    return result


# 多峰测试函数

'''
Generalized Schwefel's Problem 2.26
[-500,500]
min = -12569.5
'''

F8_lb = [-500] * 30
F8_ub = [500] * 30


def F8(X):
    result = np.sum(-X * np.sin(np.sqrt(np.abs(X))))
    return result


'''
Generalized Rastrigin's Function
[-5.12,5.12]
min = 0
'''

F9_lb = [-5.12]
F9_ub = [5.12]


def F9(X):
    dim = X.shape[0]
    O = np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X)) + 10 * dim
    return O


'''
Ackley's Function
[-32,32]
min = 0
'''

F10_lb = [-32]
F10_ub = [32]


def F10(X):
    dim = X.shape[0]
    a, b, c = 20, 0.2, 2 * np.pi
    sum_1 = -a * np.exp(-b * np.sqrt(np.sum(X ** 2) / dim))
    sum_2 = np.exp(np.sum(np.cos(c * X)) / dim)
    O = sum_1 - sum_2 + a + np.exp(1)
    return O


'''
Generalized Griewank's Function
[-600,600]
min = 0
'''

F11_lb = [-600]
F11_ub = [600]


def F11(X):
    dim = X.shape[0]
    Temp = np.arange(1, dim + 1, +1)
    result = np.sum(X ** 2) / 4000 - np.prod(np.cos(X / np.sqrt(Temp))) + 1
    return result


'''
Generalized Penalized Function 1
[-50,50]
min = 0
'''


def Ufun(X, a, k, m):
    dim = X.shape[0]
    U = np.zeros(dim)
    for i in range(len(X)):
        if X[i] > a:
            U[i] = k * ((X[i] - a) ** m)
        elif X[i] < -a:
            U[i] = k * ((-X[i] - a) ** m)
        else:
            U[i] = 0
    return U


F12_lb = [-50]
F12_ub = [50]


def F12(X):
    dim = X.shape[0]
    pi = np.pi
    sum_1 = (np.pi / dim) * (10 * ((np.sin(pi * (1 + (X[0] + 1) / 4))) ** 2)
                             + np.sum((((X[:dim - 2] + 1) / 4) ** 2) *
                                      (1 + 10 * ((np.sin(pi * (1 + (X[1:dim - 1] + 1) / 4)))) ** 2))
                             + ((X[dim - 1]) / 4) ** 2)
    sum_2 = np.sum(Ufun(X, 10, 100, 4))
    O = sum_1 + sum_2
    return O


'''
Generalized Penalized Function 2
[-50,50]
min = 0
'''

F13_lb = [-50]
F13_ub = [50]


def F13(X):
    dim = X.shape[0]
    pi = np.pi
    O = 0.1 * ((np.sin(3 * pi * X[0])) ** 2 + np.sum(
        ((X[0:dim - 2]) - 1) ** 2 * (1 + (np.sin(3 * pi * X[1:dim - 1])) ** 2)))
    +((X[dim - 1] - 1) ** 2) * (1 + (np.sin(2 * pi * X[dim - 1])) ** 2) + np.sum(Ufun(X, 5, 100, 4))
    return O


# 固定维度多峰测试函数

'''
Shekel's Foxholes Function
[-65.536,65.536]
2
min = 1
'''

F14_lb = [-65.536] * 2
F14_ub = [65.536] * 2


def F14(X):
    aS = np.array(
        [[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
         [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    bS = np.zeros(25)
    for j in range(0, 25):
        bS[j] = np.sum((X - aS[:, j]) ** 6)

    O = (1 / 500 + np.sum(1 / (np.arange(1, 25 + 1) + bS))) ** (-1)
    return O


'''
Kowalik's Function
[-5,5]
4
min = 0.0003075
'''

F15_lb = [-5] * 4
F15_ub = [5] * 4


def F15(X):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    bK = 1 / bK
    O = np.sum((aK - ((X[0] * (bK ** 2 + X[1] * bK)) / (bK ** 2 + X[2] * bK + X[3]))) ** 2)
    return O


'''
Six-Hump Camel-Back Function
[-5,5]
2
min = -1.0316285
'''

F16_lb = [-5] * 2
F16_ub = [5] * 2


def F16(X):
    O = 4 * (X[0] ** 2) - 2.1 * (X[0] ** 4) + (X[0] ** 6) / 3 + X[0] * X[1] - 4 * (X[1] ** 2) + 4 * (X[1] ** 4)
    return O


'''
Branin Function
[-5,5]
2
min = 0.398
'''

F17_lb = [-5] * 2
F17_ub = [5] * 2


def F17(X):
    pi = np.pi
    O = ((X[1]) - (X[0] ** 2) * 5.1 / (4 * (pi ** 2)) + 5 / pi * X[0] - 6) ** 2 + 10 * (1 - 1 / (8 * pi)) * np.cos(
        X[0]) + 10
    return O


'''
Goldstein-Price Function
[-2,2]
2
min = 3
'''

F18_lb = [-5] * 2
F18_ub = [5] * 2


def F18(X):
    O = (1 + ((X[0] + X[1] + 1) ** 2) * (
            19 - 14 * X[0] + 3 * (X[0] ** 2) - 14 * X[1] + 6 * X[0] * X[1] + 3 * (X[1] ** 2))) * (
                30 + (2 * X[0] - 3 * X[1]) ** 2 * (
                18 - 32 * X[0] + 12 * (X[0] ** 2) + 48 * X[1] - 36 * X[0] * X[1] + 27 * (X[1] ** 2)))
    return O


'''
Hartman's Family
[1,3]
3
min = -3.86
'''

F19_lb = [1] * 3
F19_ub = [3] * 3


def F19(X):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747],
                   [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    O = 0
    for i in range(0, 4):
        O = O - cH[i] * np.exp(-(np.sum(aH[i] * ((X - pH[i]) ** 2))))
    return O


'''
[0,1]
6
min = -3.32

'''

F20_lb = [0] * 6
F20_ub = [1] * 6


def F20(X):
    aH = np.array(
        [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.413, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    O = 0
    for i in range(0, 4):
        O = O - cH[i] * np.exp(-(np.sum(aH[i] * ((X - pH[i]) ** 2))))
    return O


'''
[0,10]
4
min = -10.1532
'''

F21_lb = [0] * 4
F21_ub = [10] * 4


def F21(X):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([[0.1], [0.2], [0.2], [0.4], [0.4],
                    [0.6], [0.3], [0.7], [0.5], [0.5]])
    O = 0
    for i in range(0, 5):
        O = O - (np.sum((X - aSH[i]) ** 2) + cSH[i]) ** (-1)
    return O


X = np.arange(4)

'''
[0,10]
4
-10.4028
'''

F22_lb = [0] * 4
F22_ub = [10] * 4


def F22(X):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([[0.1], [0.2], [0.2], [0.4], [0.4],
                    [0.6], [0.3], [0.7], [0.5], [0.5]])
    O = 0
    for i in range(0, 7):
        O = O - (np.sum((X - aSH[i]) ** 2) + cSH[i]) ** (-1)
    return O


'''
[0,10]
4
-10.5363
'''

F23_lb = [0] * 4
F23_ub = [10] * 4


def F23(X):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([[0.1], [0.2], [0.2], [0.4], [0.4],
                    [0.6], [0.3], [0.7], [0.5], [0.5]])
    O = 0
    for i in range(0, 10):
        O = O - (np.sum((X - aSH[i]) ** 2) + cSH[i]) ** (-1)
    return O
