import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    # find AT.A
    ATA = np.dot(A.T, A)

    # eigen values & vectors in descending order
    vals_eigh, vecs_eigh = np.linalg.eigh(ATA)
    idx = vals_eigh.argsort()[::-1]
    vals_eigh = vals_eigh[idx]
    vecs_eigh = vecs_eigh[:, idx]

    V = vecs_eigh.T
    s = np.sqrt(vals_eigh)

    U = A @ vecs_eigh / vals_eigh[:, None]

    # Compute left singular vectors
    U = np.zeros_like(A, dtype=float)
    for i in range(2):
        U[:, i] = A @ V[i, :] / s[i]

    return U, s, V
