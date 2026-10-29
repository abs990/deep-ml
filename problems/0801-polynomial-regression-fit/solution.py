import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    x = np.array(x)
    y = np.array(y)

    x_d = np.vander(x, N=degree + 1, increasing=True)
    return np.linalg.solve(x_d.T @ x_d, x_d.T @ y).tolist()