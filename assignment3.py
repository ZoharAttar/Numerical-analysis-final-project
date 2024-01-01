"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random
from assignment2 import Assignment2
from assignment1 import Assignment1
import matplotlib.pyplot as plt


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the integration error.
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions.

        Integration error will be measured compared to the actual value of the
        definite integral.

        Note: It is forbidden to call f more than n times.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        if n-2 == 0:
            ans = ((f(a)+f(b))/2)*(b-a)
            return np.float32(ans)

        update_n = n-2
        odds = 0
        even = 0
        if n % 2 != 0:
            update_n = n-1
        h = (b-a)/update_n
        for i in range(1, update_n):
            x = a + (i*h)
            if i % 2 == 0:
                even += f(x)
            else:
                odds += f(x)
        ans = (h / 3) * (f(a) + 4*odds + 2*even + f(b))
        result = np.float32(ans)

        return result

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """
        ass2 = Assignment2()
        points = ass2.intersections(f1, f2, 1, 100)
        if len(points) < 2:
            return
        i = 0
        n = 100
        area = 0

        while i < len(points)-1:
            if f1(points[i]+0.01) < f2(points[i]+0.01):
                f = lambda x: f2(x) - f1(x)
                area += self.integrate(f, points[i], points[i+1], n)
            else:
                f = lambda x: f1(x) - f2(x)
                area += self.integrate(f, points[i], points[i + 1], n)
            i += 1

        result = np.float32(area)

        return result

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
import math


class TestAssignment3(unittest.TestCase):

    # ass3 = Assignment3()
    # f1 = np.poly1d([-1, 0, 1])
    # r = ass3.integrate(f1, -1, 1, 10)
    # print(r)

    def test_integrate_float32(self):
        ass3 = Assignment3()
        # f3 = lambda x: pow(x, 2) - 3 * x + 2
        # f4 = lambda x: -pow(x, 2) + 5
        # r4 = ass3.areabetween(f3, f4)
        # print('24.343224 polys = ', r4)
        T = time.time()


        f1 = lambda x: np.sin(np.log(x))
        f2 = lambda x: ((np.sin(x))/x) - 0.1
        r = ass3.areabetween(f1,f2)
        print('14.0539 sin+sin = ', r)

        self.assertEqual(r.dtype, np.float32)
        f3= lambda x: pow(x, 2) - 3 * x + 2
        r2 = ass3.areabetween(f1, f3)
        print('0.731257 poly+sin(log) = ', r2)
        self.assertEqual(r2.dtype, np.float32)
        T = time.time() - T
        print('time - ', np.float32(T))


    # def test_integrate_float32(self):
    #     ass3 = Assignment3()
    #     f1 = lambda x: np.sin(pow(x, 2))
    #     r = ass3.integrate(f1, 5, 1.5, 4)
    #     print('sinx^2 area = ', r)
    #
    #     self.assertEqual(r.dtype, np.float32)

    def test_integrate_float302(self):
        ass3 = Assignment3()
        f1 = lambda x: (np.sin(1/(pow(x, 2))))
        r = ass3.integrate(f1, 5, 1.5, 4)
        print('1/sinx^2 area = ', r)

        self.assertEqual(r.dtype, np.float32)

    def test_integrate_float342(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)
        print(r)

        self.assertEqual(r.dtype, np.float32)

    def test_integrate_float321(self):
        ass3 = Assignment3()
        f1 = lambda x: pow(x, 2) - 3 * x + 2
        T = time.time()
        r = ass3.integrate(f1, 2, 10, 10)
        T = time.time() - T
        print('f2 area = ', r)
        print(('time: ', np.float32(T)))

        self.assertEqual(r.dtype, np.float32)

    def test_integrate_float31(self):
        ass3 = Assignment3()
        f1 = lambda x: 1 - x - 2 * (x ** 2) + x ** 3
        r = ass3.integrate(f1, -0.802, 2.247, 1499)
        print('f2 area = ', r)

        self.assertEqual(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_integrate(self):
        ass3 = Assignment3()
        funcs = [
            lambda x: 5 * (math.e ** (-(x ** 2))),
            lambda x: math.sqrt(1 - (x ** 4)),
            lambda x: math.log(6 * math.log(x, math.e), math.e),
            lambda x: math.e ** (-x) / x,
            lambda x: (x ** (math.e - 1)) * (math.e ** (-x)),
            lambda x: 1 - x - 2 * (x ** 2) + x ** 3,
            lambda x: 1 - 999999 * (x ** 4),
            lambda x: np.sin(x) * ((3 - (x ** 2) + x ** 3) ** 0.5),
        ]
        param = [
            (0, 4, 10),
            (0, 1, 109),
            (2, 5, 55),
            (2, 5, 11),
            (2, 5, 1000),
            (-0.802, 2.247, 1499),
            (-0.0316, 0.0316, 151),
            (0, 2 * np.pi, 20)
        ]
        sol = [
            4.4311345589,
            0.8740191847,
            5.8983046955,
            0.04775,
            0.8035201908,
            -0.7915225965,
            0.05059638959,
            -13.931856598865,
        ]
        error = 0.01
        i = 0
        print('\n==========================================================================================')
        print(f'                                        RESULTS')
        total_err = 0
        total_time = 0
        for f in funcs:
            S = sol[i]
            T = time.time()
            SS = ass3.integrate(f, param[i][0], param[i][1], param[i][2])
            T = time.time() - T
            total_time += T
            self.assertLessEqual(abs(S - SS), error)
            print(
                f'\nFUNCTION #{i + 1}..\n     S = {S}\n     SS = {SS}\n     Error: {abs(S - SS)}\n     Time: {round(T, 3)} (s)')
            i += 1
            total_err += abs(S - SS)
        print('\n==========================================================================================')
        print(f'TOTAL ERROR: {total_err}')
        print(f'TOTAL TIME:  {round(total_time, 5)}')
        print('==========================================================================================')

    def test_area_between(self):
        ass3 = Assignment3()
        funcs = [
            (lambda x: 5 - (-5 + x) ** 2, lambda x: 2 * ((-5 + x) ** 2) - 5),
            (lambda x: 4 - (-7 + x) ** 6, lambda x: (-7 + x) ** 3),
        ]
        sol = [
            24.343224,
            8.850282646,
        ]

        print('\n==========================================================================================')
        print(f'                                        RESULTS')
        total_err = 0
        total_time = 0
        for i in range(len(funcs)):
            S = sol[i]
            T = time.time()
            SS = ass3.areabetween(funcs[i][0], funcs[i][1])
            T = time.time() - T
            total_time += T
            print(
                f'\nFunction number {i + 1}..\n     S = {S}\n     SS = {SS}\n     Error: {abs(S - SS)}\n     Time: {round(T, 3)} (s)\n')
            i += 1
            total_err += abs(S - SS)
        print(f'TOTAL ERROR => {total_err}')
        print(f'TOTAL TIME  => {round(total_time, 5)}')
        SS = ass3.areabetween(lambda x: 5, lambda x: 6)
        self.assert_(SS is np.nan, "Failed case when function not intersecting in [1,100] or less than 2 roots")
        print('\n==========================================================================================')


if __name__ == "__main__":
    unittest.main()
