"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable
from assignment1 import Assignment1


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """
        ranges = np.arange(a, b+maxerr/2, maxerr)

        f = lambda x: f1(x) - f2(x)
        delta_x = 10**(-25)
        if not (type(f1) == np.poly1d) or not (type(f2) == np.poly1d):
            df = lambda x: (f(x+delta_x)-f(x))/delta_x
        else:
            df = np.polyder(f1 - f2)
        X = []

        def newton_raphson(g: callable, dg, x1: float, x2: float) -> float:
            x = (x2 - x1) / 2
            counter = 0

            if abs(g(x1)) <= maxerr:
                return x1
            elif abs(g(x2)) <= maxerr:
                return x2

            while abs(g(x)) >= maxerr:
                if counter > 20:
                    return
                if dg(x) == 0:
                    h = g(x) / 0.0000000001
                else:
                    h = g(x) / dg(x)
                x = x - h
                counter += 1
                if not x1 < x < x2:
                    x = random.uniform(x1, x2)
            return x

        for i in range(len(ranges) - 1):
            start = ranges[i]
            end = ranges[i+1]
            if f(start) * f(end) <= 0:
                root = newton_raphson(f, df, ranges[i], ranges[i + 1])
                if root != None:
                    if len(X) > 0:
                        if abs(root - X[-1]) > maxerr:
                                X.append(root)
                    else:
                        X.append(root)
        return X

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
import math
import numpy as np


class TestAssignment2(unittest.TestCase):

    # def test_1(self):
    #     ass2 = Assignment2()
    #
    #     f1 = np.poly1d([1, 0, 0])
    #     f2 = np.poly1d([1])
    #
    #     X = ass2.intersections(f1, f2, -1, 1)
    #     print('test1: ', X)
    #
    #     for x in X:
    #         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    # def test_1(self):
    #     ass2 = Assignment2()
    #
    #     f1 = lambda x: np.sin(pow(x, 2))
    #     f2 = lambda x: pow(x, 2) - 3 * x + 2
    #
    #     X = ass2.intersections(f1, f2, 0.5, 2)
    #     print('test1: ', X)
    #
    #     for x in X:
    #         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_2(self):
        ass2 = Assignment2()

        f1 = lambda x: np.sin(pow(x, 2))
        f2 = lambda x: np.sin(np.log(x))

        X = ass2.intersections(f1, f2, 1, 4)
        print('test1: ', X)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    # def test_sqr(self):
    #
    #     ass2 = Assignment2()
    #
    #     f1 = lambda x: math.e ** x
    #     f2 = np.poly1d([0, 1, 2])
    #     X = ass2.intersections(f1, f2, -2, 2, maxerr=0.001)
    #     print('test_sqr: ', X)
    #
    #     for x in X:
    #         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
    #
    # def test_sqr1(self):
    #     ass2 = Assignment2()
    #
    #     f1 = lambda x: math.sin(x)
    #     f2 = np.poly1d([0, 0, 1])
    #     X = ass2.intersections(f1, f2, -5, 10, maxerr=0.001)
    #     print('test_sqr1: ', X)
    #
    #     for x in X:
    #         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
    #
    #
    # def test_sqr2(self):
    #
    #     ass2 = Assignment2()
    #
    #     f1 = np.poly1d([-1, 0, 1])
    #     f2 = np.poly1d([1, 0, -1])
    #
    #     X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
    #     print('test_sqr2: ', X)
    #     for x in X:
    #         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
    #
    # def test_poly(self):
    #
    #     ass2 = Assignment2()
    #
    #     f1, f2 = randomIntersectingPolynomials(10)
    #
    #     X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
    #     print('test_poly: ', X)
    #     for x in X:
    #         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    # def test_wierdies(self):
    #     e = np.e
    #     funcs = [
    #         lambda x: 1 - x - 2 * (x ** 2) + x ** 3,
    #         lambda x: np.sin(x),
    #         lambda x: math.log(math.sin(x), e),
    #         lambda x: np.sin(x) / (x ** 2),
    #         lambda x: (7 * np.cos(18 * x + e) + 16 * (x ** 2)) / 5,
    #         lambda x: 3 * np.sin(9 * x) - np.cos(x / 6) * x + x + 7,
    #         lambda x: (e ** x) * (np.log(7 * x)) + (e ** x) / x - 45 * (x ** 2),
    #     ]
    #     parameters = [
    #         (lambda x: 0, -1, 2.3, 0.001, 3),
    #         (lambda x: 0.05 * x, -4, 10, 0.001, 5),
    #         (lambda x: np.sin(100 * x), 0.4, 0.6, 0.001, 6),
    #         (lambda x: np.sin(np.log(x ** 6)) / x, -5, -0.4, 0.001, 6),
    #         (lambda x: 0, -20, 20, 0.00001, 6),
    #         (lambda x: 0, -20, 0, 0.0001, 9),
    #         (lambda x: 0, 0.2, 0.8, 0.00001, 1)
    #     ]
    #     print('\n==========================================================================================')
    #     print(f'                                        RESULTS')
    #     p = 0
    #     total_T = 0
    #     total_E = 0
    #     ass2 = Assignment2()
    #     for f1 in funcs:
    #         f2 = parameters[p][0]
    #         a = parameters[p][1]
    #         b = parameters[p][2]
    #         err = parameters[p][3]
    #         roots = parameters[p][4]
    #         T = time.time()
    #         X = ass2.intersections(f1, f2, a, b, err)
    #         T = time.time() - T
    #         print(f'\nFUNCTION #{p + 1}:\nNumber of roots: {len(X)}\{roots}\n{X}\nTime: {round(T, 3)} (s)')
    #         root_num = 1
    #         self.assertEqual(roots, len(X))
    #         for x in X:
    #             self.assertGreaterEqual(0.00001, abs(f1(x) - f2(x)))
    #             total_E += abs(f1(x) - f2(x))
    #             print(f'    Error of root #{root_num}: {abs(f1(x) - f2(x))}')
    #             root_num += 1
    #         p += 1
    #         total_T += T
    #
    #     print('\n==========================================================================================')
    #     print(f'TOTAL ERROR: {total_E}')
    #     print(f'TOTAL TIME:  {total_T}')
    #     print('==========================================================================================')


if __name__ == "__main__":
    unittest.main()
