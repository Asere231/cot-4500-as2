# src/test/test_assignment_2.py

import unittest
import numpy as np
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.main.assignment_2 import (
    nevilles_method,
    newton_forward_difference,
    newton_interpolation,
    hermite_interpolation,
    cubic_spline
)

class TestAssignment2(unittest.TestCase):
    def test_nevilles_method(self):
        x = np.array([3.6, 3.8, 3.9])
        y = np.array([1.675, 1.436, 1.318])
        result = nevilles_method(x, y, 3.7)
        self.assertIsNotNone(result)
        
    def test_newton_forward_difference(self):
        x = np.array([7.2, 7.4, 7.5, 7.6])
        y = np.array([23.5492, 25.3913, 26.8224, 27.4589])
        result = newton_forward_difference(x, y)
        self.assertEqual(result.shape, (4, 4))
        
    def test_newton_interpolation(self):
        x = np.array([7.2, 7.4, 7.5, 7.6])
        y = np.array([23.5492, 25.3913, 26.8224, 27.4589])
        result = newton_interpolation(x, y, 7.3)
        self.assertIsNotNone(result)
        
    def test_hermite_interpolation(self):
        x = np.array([3.6, 3.8, 3.9])
        y = np.array([1.675, 1.436, 1.318])
        derivatives = np.array([-1.195, -1.188, -1.182])
        result = hermite_interpolation(x, y, derivatives)
        self.assertEqual(result.shape, (6, 6))
        
    def test_cubic_spline(self):
        x = np.array([2, 5, 8, 10])
        y = np.array([3, 5, 7, 9])
        A, b, x = cubic_spline(x, y)
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(b.shape, (2,))
        self.assertEqual(x.shape, (2,))

if __name__ == '__main__':
    unittest.main()
