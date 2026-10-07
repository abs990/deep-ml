
import numpy as np

def matrix_image(A):
	M, N = A.shape

	# RREF
	A_temp = A.astype(float).copy()
	rank = 0
	tol = 1e-10

    ## keep finding pivot row and eliminate entries below it
	for col in range(N):
        ### Find pivot ONLY among rows that haven't already been used
		pivot = rank + np.argmax(np.abs(A_temp[rank:, col]))
		if abs(A_temp[pivot, col]) < tol:
			continue

        ## Move pivot row into current rank position
		A_temp[[rank, pivot]] = A_temp[[pivot, rank]]

        ### Eliminate entries below pivot
		factors = A_temp[rank + 1:M, col] / A_temp[rank, col]
		A_temp[rank + 1:M, col:N] -= (factors[:, None] * A_temp[rank, col:N])

        ### update rank
		rank += 1
		if rank == M:
			break

	# identify pivot column indices
	pivot_cols = np.array([nz[0] for row in A_temp if(nz := np.flatnonzero(np.abs(row) > tol)).size])
	
	return A[:, pivot_cols]
