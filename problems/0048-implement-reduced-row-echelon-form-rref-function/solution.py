import numpy as np

def rref(matrix):
	matrix = matrix.astype(float).copy()
	M, N = matrix.shape
	rank = 0
	tol = 1e-10
	
	# keep finding pivot row and eliminate entries below it
	for col in range(N):
		# Find pivot ONLY among rows that haven't already been used
		pivot = rank + np.argmax(np.abs(matrix[rank:, col]))
		if abs(matrix[pivot, col]) < tol:
			continue
			
		# Move pivot row into current rank position
		matrix[[rank, pivot]] = matrix[[pivot, rank]]

		# Scale pivot row
		matrix[rank, :] /= matrix[rank, col]
		
		# Eliminate entries above and below pivot
		indices = np.arange(M)
		indices = indices[indices != rank]
		factors = matrix[indices, col] / matrix[rank, col]
		matrix[indices, col:N] -= (factors[:, None] * matrix[rank, col:N])
		
		# update rank
		rank += 1
		if rank == M:
			break

	return matrix
