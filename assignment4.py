"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random
import math


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        self.maxtime_dict = {}

        def getMatrixMinor(arr, i, j):
            return np.delete(np.delete(arr, i, axis=0), j, axis=1)

        def getMatrixDeternminant(m):
            # base case for 2x2 matrix
            if len(m) == 2:
                return m[0][0] * m[1][1] - m[0][1] * m[1][0]

            determinant = 0
            for c in range(len(m)):
                determinant += ((-1) ** c) * m[0][c] * getMatrixDeternminant(getMatrixMinor(m, 0, c))
            return determinant

        def getMatrixInverse(m):
            determinant = getMatrixDeternminant(m)
            # special case for 2x2 matrix:
            if len(m) == 2:
                return np.array([[m[1][1] / determinant, -1 * m[0][1] / determinant],
                         [-1 * m[1][0] / determinant, m[0][0] / determinant]])

            # find matrix of cofactors
            cofactors = []
            for r in range(len(m)):
                cofactorRow = []
                for c in range(len(m)):
                    minor = getMatrixMinor(m, r, c)
                    cofactorRow.append(((-1) ** (r + c)) * getMatrixDeternminant(minor))
                cofactors.append(cofactorRow)
            cofactors = np.transpose(cofactors)
            for r in range(len(cofactors)):
                for c in range(len(cofactors)):
                    cofactors[r][c] = cofactors[r][c] / determinant
            return np.array(cofactors)

        def which_n(n: int) -> (int, np.ndarray, np.ndarray):
            self.n = n
            self.t = 1 / self.n
            self.M = np.array([[0, 0, 0, 1], [0, 0, 3, -3], [0, 3, -6, 3], [1, -3, 3, -1]])

            self.T = np.zeros((self.n, 4))
            for i in range(1, self.n):
                row = np.array([1, self.t * i, (self.t * i) ** 2, (self.t * i) ** 3])
                self.T[i - 1] = row
                if i == self.n - 1:
                    self.T[i] = np.array([1, 1, 1, 1])

            self.TT = np.transpose(self.T)
            self.M_INV = np.array([[1, 1, 1, 1], [1, 2 / 3, 1 / 3, 0], [1, 1 / 3, 0, 0], [1, 0, 0, 0]])
            self.TT_T_INV = getMatrixInverse(self.TT.dot(self.T))
            self.C_without_P = (self.M_INV.dot(self.TT_T_INV)).dot(self.TT)
            return self.n, self.C_without_P, self.M

        self.maxtime_dict[15] = which_n(900)
        self.maxtime_dict[10] = which_n(100)
        self.maxtime_dict[5] = which_n(10)


    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        if maxtime >= 15:
            n, C_without_P, M = self.maxtime_dict[15]
        elif maxtime >= 10:
            n, C_without_P, M = self.maxtime_dict[10]
        else:
            n, C_without_P, M = self.maxtime_dict[5]

        if b < a:
            a, b = b, a
        x_p = np.linspace(a, b, n)
        y_p = np.array([f(x) for x in x_p])

        P = np.zeros((n, 2))

        for k in range(len(x_p)):
            P[k][0] = x_p[k]
            P[k][1] = y_p[k]

        C = np.array(C_without_P.dot(P))

        def distance(p1, p2):
            return math.dist(p1,p2)

        def finding_distance(i: int, d: float) -> float:
            if i == len(x_p) - 1:
                return 0
            return finding_distance(i + 1, distance([x_p[i+1], x_p[i]], [y_p[i+1], y_p[i]])) + distance([x_p[i+1], x_p[i]], [y_p[i+1], y_p[i]])

        def finding_distance_to_x(i: int, d: float, end: int) -> float:
            if i == end:
                return 0
            return finding_distance_to_x(i + 1, distance([x_p[i+1], x_p[i]], [y_p[i+1], y_p[i]]), end) + distance([x_p[i+1], x_p[i]], [y_p[i+1], y_p[i]])

        def finding_t(x: float) -> float:
            d1 = finding_distance(0, 0)
            j = 0
            while j < len(x_p):
                if x >= x_p[j] and x <= x_p[j + 1]:
                    fx = (x - x_p[j]) * ((y_p[j + 1] - y_p[j]) / (x_p[j + 1] - x_p[j])) + y_p[j]
                    d_to_x = finding_distance_to_x(0, 0, j) + ((x - x_p[j]) ** 2 + ((fx - y_p[j]) ** 2)) ** 0.5
                    t = d_to_x / d1
                    return t
                j += 1

        def bezier(my_t: float) -> float:
            T = np.array([1, my_t, my_t ** 2, my_t ** 3])
            return (T.dot(M)).dot(C)

        result = lambda x: bezier(finding_t(x))[1]

        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
from commons import *
import threading
import _thread as thread


class TestAssignment4(unittest.TestCase):


    # def test_return(self):
    #     f = lambda x: log(log(x))
    #     nf = NOISY(1)(f)
    #     ass4 = Assignment4A()
    #     T = time.time()
    #     ff = ass4.fit(f=nf, a=5, b=10, d=10, maxtime=20)
    #     T = time.time() - T
    #     mse = 0
    #     for x in np.linspace(5, 10, 1000):
    #         self.assertNotEqual(f(x), nf(x))
    #         mse += (f(x) - ff(x)) ** 2
    #     mse = mse / 1000
    #     print('mas: ', mse)

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))
    #     ass4 = Assignment4A()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
    #     T = time.time() - T
    #     self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEqual(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)

    # T = time.time()
    # f = np.poly1d([1, -1, 33])
    # ass4 = Assignment4A()
    # ans = ass4.fit(f, -1, 1, 10, 0.5)
    # print('ans : 0.5 , y(0) = ', ans(0))
    # T = time.time() - T
    # print('time:', T)
    # T = time.time()
    # ans1 = ass4.fit(f, -1, 1, 10, 0.05)
    # print('ans1 : 0.05 , y(0) = ', ans(0))
    # T = time.time() - T
    # print('time:', T)
    # T = time.time()
    # ans2 = ass4.fit(f, -1, 1, 10, 0.005)
    # print('ans2 : 0.005 , y(0) = ', ans(0))
    # T = time.time() - T
    # print('time:',  T)


if __name__ == "__main__":
    unittest.main()
