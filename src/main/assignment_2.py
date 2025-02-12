# src/main/assignment_2.py

import numpy as np

def nevilles_method(x_points, y_points, x):
    """
    Implements Neville's method for polynomial interpolation
    """
    n = len(x_points)
    p = np.zeros((n, n))
    
    # Fill in the first column with y values
    for i in range(n):
        p[i, 0] = y_points[i]
    
    # Fill in the rest of the matrix
    for j in range(1, n):
        for i in range(n - j):
            p[i, j] = ((x - x_points[i + j]) * p[i, j - 1] - 
                       (x - x_points[i]) * p[i + 1, j - 1]) / \
                      (x_points[i] - x_points[i + j])
    
    return p[0, n-1]

def newton_forward_difference(x_points, y_points):
    """
    Implements Newton's forward difference method
    """
    n = len(x_points)
    f = np.zeros((n, n))
    
    # Fill in the first column with y values
    f[:, 0] = y_points
    
    # Calculate the forward differences
    for j in range(1, n):
        for i in range(n - j):
            f[i, j] = f[i + 1, j - 1] - f[i, j - 1]
    
    return f

def newton_interpolation(x_points, y_points, x):
    """
    Implements Newton's interpolation method
    """
    n = len(x_points)
    f = newton_forward_difference(x_points, y_points)
    
    # Calculate the interpolation
    result = f[0, 0]
    xterm = 1
    for j in range(1, n):
        xterm *= (x - x_points[j - 1])
        result += (f[0, j] * xterm) / np.prod([x_points[k] - x_points[0] for k in range(1, j + 1)])
    
    return result

def hermite_interpolation(x_points, y_points, derivatives):
    """
    Implements Hermite interpolation
    """
    n = len(x_points)
    matrix = np.zeros((2 * n, 2 * n))
    
    # Fill in the x values and function values
    z = np.zeros(2 * n)
    for i in range(n):
        z[2 * i] = x_points[i]
        z[2 * i + 1] = x_points[i]
        matrix[2 * i, 0] = y_points[i]
        matrix[2 * i + 1, 0] = y_points[i]
        matrix[2 * i + 1, 1] = derivatives[i]
        if i != 0:
            matrix[2 * i, 1] = (matrix[2 * i, 0] - matrix[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])
    
    # Fill in the divided differences
    for j in range(2, 2 * n):
        for i in range(2 * n - j):
            matrix[i, j] = (matrix[i + 1, j - 1] - matrix[i, j - 1]) / (z[i + j] - z[i])
    
    return matrix

def cubic_spline(x_points, y_points):
    """
    Implements cubic spline interpolation
    """
    n = len(x_points) - 1
    h = np.diff(x_points)
    
    # Create the matrix A
    A = np.zeros((n-1, n-1))
    b = np.zeros(n-1)
    
    # Fill in the tridiagonal matrix
    for i in range(n-1):
        if i > 0:
            A[i, i-1] = h[i]
        A[i, i] = 2 * (h[i] + h[i+1])
        if i < n-2:
            A[i, i+1] = h[i+1]
            
        b[i] = 3 * ((y_points[i+2] - y_points[i+1]) / h[i+1] - 
                    (y_points[i+1] - y_points[i]) / h[i])
    
    # Solve the system Ax = b
    x = np.linalg.solve(A, b)
    
    return A, b, x

if __name__ == "__main__":
    # Question 1
    x1 = np.array([3.6, 3.8, 3.9])
    y1 = np.array([1.675, 1.436, 1.318])
    result1 = nevilles_method(x1, y1, 3.7)
    print(f"1. f(3.7) using Neville's method: {result1}\n")
    
    # Question 2
    x2 = np.array([7.2, 7.4, 7.5, 7.6])
    y2 = np.array([23.5492, 25.3913, 26.8224, 27.4589])
    forward_diff = newton_forward_difference(x2, y2)
    print("2. Newton's forward difference table:")
    print(forward_diff)
    print()
    
    # Question 3
    result3 = newton_interpolation(x2, y2, 7.3)
    print(f"3. f(7.3) using Newton's interpolation: {result3}\n")
    
    # Question 4
    x4 = np.array([3.6, 3.8, 3.9])
    y4 = np.array([1.675, 1.436, 1.318])
    derivatives = np.array([-1.195, -1.188, -1.182])
    hermite_matrix = hermite_interpolation(x4, y4, derivatives)
    print("4. Hermite interpolation matrix:")
    print(hermite_matrix)
    print()
    
    # Question 5
    x5 = np.array([2, 5, 8, 10])
    y5 = np.array([3, 5, 7, 9])
    A, b, x = cubic_spline(x5, y5)
    print("5. Cubic spline results:")
    print("Matrix A:")
    print(A)
    print("\nVector b:")
    print(b)
    print("\nVector x:")
    print(x)
