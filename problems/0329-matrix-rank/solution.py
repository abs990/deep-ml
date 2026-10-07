import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    # Your code here
    A = A.astype(float).copy()
    M, N = A.shape
    rank = 0

    # keep finding pivot row and eliminate entries below it
    for col in range(N):
        # Find pivot ONLY among rows that haven't already been used
        pivot = rank + np.argmax(np.abs(A[rank:, col]))
        if abs(A[pivot, col]) < tol:
            continue

        # Move pivot row into current rank position
        A[[rank, pivot]] = A[[pivot, rank]]

        # Eliminate entries below pivot
        factors = A[rank + 1:M, col] / A[rank, col]
        A[rank + 1:M, col:N] -= (
            factors[:, None] * A[rank, col:N]
        )

        # update rank
        rank += 1
        if rank == M:
            break
    
    return rank