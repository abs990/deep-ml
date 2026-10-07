import numpy as np

def cramers_rule(A, b):
    A = np.array(A)
    b = np.array(b)

    A_det = np.linalg.det(A)

    if A_det == 0:
        return -1

    n = A.shape[0]
    x = np.zeros_like(b, dtype=float)

    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / A_det

    return x