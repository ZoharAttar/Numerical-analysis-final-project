"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape
import matplotlib.pyplot as plt
from assignment4 import Assignment4A


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, area):
        self.shape_area = area

    def sample(self):
        pass

    def contour(self, n: int):
        return

    def area(self):
        return self.shape_area


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """

        def area_n(m: int) -> float:
            points = contour(m)
            area0 = 0
            i = 0
            while i < len(points) - 1:
                area0 += (points[i + 1][0] - points[i][0]) * (((points[i + 1][1]) + (points[i][1])) / 2)
                i += 1
            return area0

        n = 500
        area1 = area_n(n)
        area2 = area_n(4 * n)
        while True:
            if area2 - area1 < maxerr:
                return np.float32(abs(area2))
            area1 = area2
            area2 = area_n(2 * n)

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        def left_most(points: list) -> int:
            min = 0
            for i in range(1, len(points)):
                if points[i][0] < points[min][0]:
                    min = i
                elif points[i][0] == points[min][0]:
                    if points[i][1] > points[min][1]:
                        min = i

            return min

        def direction(a: list, b: list, c: list) -> int:
            orientation = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

            if orientation == 0:
                return 0
            elif orientation > 0:
                return 1
            else:
                return 2

        def convex(points: list) -> list:
            convex_hull = []
            if len(points) < 3:
                return
            current_point = left_most(points)
            hull = []
            a = current_point
            b = 0
            while True:
                hull.append(a)
                b = (a + 1) % len(points)

                for i in range(len(points)):
                    if direction(points[a], points[i], points[b]) == 2:
                        b = i
                a = b
                if a == current_point:
                    break
            for n in hull:
                convex_hull.append([points[n][0], points[n][1]])
            return hull

        def sortPoints(points: list) -> np.ndarray:
            x = np.array(points)
            sum_x = 0
            sum_y = 0
            for i in x:
                sum_x += i[0]
                sum_y += i[1]
            avg = (sum_x / len(x))
            xs = x
            xs[:, 0] -= avg
            xs[:, 1] -= avg

            x_sort = xs[np.angle((xs[:, 0] + 1j * xs[:, 1])).argsort()]

            x_sort[:, 0] += avg
            x_sort[:, 1] += avg

            return x_sort

        def fit_line(segment:np.ndarray) -> list:
            xs = segment[:, 0]
            ys = segment[:, 1]
            res = []
            pol = np.polyfit(xs, ys, 3)
            f = np.poly1d(pol)
            space = ((segment[-1][0] - segment[0][0]) / (0.5 * len(segment)))
            spaces = np.arange(segment[0][0], segment[-1][0], space)
            for s in spaces:
                res.append([s, f(s)])

            return res

        data = []
        for i in range(500):
            point = sample()
            data.append(point)

        sorted_data = sortPoints(data).tolist()
        hull_indexs = convex(sorted_data)

        i = 0
        left = False
        turn_points_index = [hull_indexs[0]]
        while i < len(hull_indexs) - 1:
            if sorted_data[hull_indexs[i + 1]][0] <= sorted_data[hull_indexs[i]][0] and not left:
                left = True
                turn_points_index.append(hull_indexs[i])
                continue
            elif sorted_data[hull_indexs[i + 1]][0] >= sorted_data[hull_indexs[i]][0] and left:
                left = False
                turn_points_index.append(hull_indexs[i])
                continue
            i += 1

        index = 0
        ans = []
        while index < len(turn_points_index) - 1:
            if turn_points_index[index] < turn_points_index[index + 1]:
                segment0 = sorted_data[turn_points_index[index]:turn_points_index[index + 1]]
            else:
                temp1 = sorted_data[turn_points_index[index]:]
                temp1 += sorted_data[:turn_points_index[index + 1]]
                segment0 = temp1
            segment0 = np.array(segment0)
            ans += fit_line(segment0)
            index += 1
        if turn_points_index[index] < turn_points_index[0]:
            ans += fit_line(np.array((sorted_data[turn_points_index[index]:turn_points_index[0]])))
        else:
            temp3 = sorted_data[turn_points_index[index]:]
            temp3 += sorted_data[:turn_points_index[0]]
            a = temp3
            ans += fit_line(np.array(a))

        area = 0
        i = 0
        while i < len(ans) - 1:
            area += (ans[i + 1][0] - ans[i][0]) * (((ans[i + 1][1]) + (ans[i][1])) / 2)
            i += 1

        result = MyShape(abs(area))

        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
from commons import *
import threading
import _thread as thread



class TestAssignment5(unittest.TestCase):

    # def test_delay(self):
    #     ass5 = Assignment5()
    #     circ1 = Circle(1, 1, 2, 0)
    #     ff = ass5.area(circ1)
    #     print('area1 = ', ff)

    def test_L_test(self):
        L = shape5().sample
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=L, maxtime=20)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #
    #     def sample():
    #         time.sleep(7)
    #         return circ()
    #
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=sample, maxtime=5)
    #     T = time.time() - T
    #     self.assertTrue(isinstance(shape, AbstractShape))
    #     self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
