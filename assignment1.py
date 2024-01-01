"""
In this assignment you should interpolate the given function.
"""
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import math



class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        if n == 1:
            return lambda x: x

        # plt.style.use('seaborn-poster')

        def divided_diff(x, y):
            """
            function to calculate the divided
            differences table
            """
            n = len(y)
            coef = np.zeros([n, n])
            # the first column is y
            coef[:, 0] = y

            for j in range(1, n):
                for i in range(n - j):
                    coef[i][j] = \
                        (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

            return coef

        def newton_poly(coef, x_data, x):
            """
            evaluate the newton polynomial
            at x
            """
            n = len(x_data) - 1
            p = coef[n]
            for k in range(1, n + 1):
                p = coef[n - k] + (x - x_data[n - k]) * p
            return p

        x = np.linspace(a, b, n)
        y = np.array([f(s) for s in x])
        # get the divided difference coef
        a_s = divided_diff(x, y)[0, :]

        # evaluate on new data points
        h = (b - a) / 6000
        x_new = np.arange(a, b + h, h)
        y_new = newton_poly(a_s, x, x_new)

        def g(x):
            for j in range(len(x_new)):
                if x >= x_new[j] and x <= x_new[j+1]:
                    return y_new[j]

        result = lambda x: g(x)

        return result


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)
            ff = ass1.interpolate(f, -10, 10, 20)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    # def test_with_poly_restrict(self):
    #     ass1 = Assignment1()
    #     a = np.random.randn(5)
    #     f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
    #     ff = ass1.interpolate(f, -10, 10, 10)
    #     xs = [0,1,2,3,4,5,6,7]
    #     for x in xs:
    #         yy = ff(x)

    # def test_with_poly_restrict(self):
    #     ass1 = Assignment1()
    #     f = np.poly1d([1, -1, 0])
    #     ff = ass1.interpolate(f, -5, 15, 20)
    #     print(f(-1) - ff(-1))
    #
    #     f1 = lambda x: pow(x, 2) - 3 * x + 2
    #     ff1 = ass1.interpolate(f1, 0, 5, 10)
    #     print('error: ', (f1(2) - ff1(2)))
    #     print('f1(2) = ', f1(2))
    #     print('ff1(2) = ', ff1(2))

    def test_poly_func(self):
        ass1 = Assignment1()

        mean_err = 0
        funcs= [
            lambda x: -x ** 2 + 25,  # 1
            lambda x: -x ** 4 + x ** 2 - x + 15,  # 2
            lambda x: 1 - x - 2 * (x ** 2) + x ** 3,  # 3
            lambda x: np.log(x + 151.1),  # 4
            lambda x: np.sin(x ** 3),  # 5
            lambda x: math.atan(np.sin(x)) * x,  # 6
            lambda x: np.e ** (x ** 2),  # 7
            lambda x: np.sin(x ** 2) * (np.e ** (x ** 3))  # 8
            ]
        cases = [  # (a, b, n)
            (-10, 10, 10),
            (-10, 10, 100),
            (-5, 29, 100),
            (-150, 150, 100),
            (-150, 150, 1000)
        ]
        print(f'\nTest for polynomial functions\n============================================================\n')
        sum_total = 0
        for j in range(len(cases)):
            print(f'******** CASE NUMBER {j + 1} ********\n        '
                  f'range=[{cases[j][0]},{cases[j][1]}] n={cases[j][2]}\n')
            num = 0
            total_error = 0
            total_time = 0
            for f in funcs:
                T = time.time()
                num += 1
                for i in tqdm(range(100)):
                    ff = ass1.interpolate(f, cases[j][0], cases[j][1], cases[j][2])
                    xs = np.random.random(200)
                    err = 0
                    for x in xs:
                        yy = ff(x)
                        y = f(x)
                        err += abs(y - yy)
                    err = err / 200
                    mean_err += err
                T = time.time() - T
                mean_err = mean_err / 100
                total_time += T
                total_error += mean_err
                print(f'Function #{num}')
                print(f'    Time:  {round(T, 5)}')
                print(f'    Error: {round(mean_err, 5)}')
            print(f'\ntotal_time: {round(total_time, 2)}')
            print(f'total_error: {round(total_error, 5)}')
            print('\n')
            sum_total += total_error
        print(f'SUMMED ERROR FOR ALL CASES: {round(sum_total, 5)}')

    def test_with_trig(self):
        T = time.time()
        ass1 = Assignment1()

        mean_err = 0
        funcs = [
            lambda x: math.atan(x),
            lambda x: math.sin(x) / x,
            lambda x: math.sin(x) * math.sin(x),
            lambda x: math.pow(math.sin(x), 15)
        ]
        i = 0
        for f in funcs:
            for i in tqdm(range(100)):
                ff = ass1.interpolate(f, -10, 10, 100)

                xs = np.random.random(1000)
                err = 0
                for x in xs:
                    yy = ff(x)
                    y = f(x)
                    err += abs(y - yy)

                err = err / 1000
                mean_err += err
                print(f'function number {i}, error={err}')
                i += 1
        mean_err = mean_err / 100

        T = time.time() - T
        print(f'\nTest for polynomial functions')
        print(f'Time taken: {T}')
        print(f'Error: {mean_err}')




if __name__ == "__main__":
    unittest.main()
