import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	m, n = A.shape
	
	# to store solution
	x = np.zeros_like(b)
	
	# no need to do process for final row
	row_end = m - 1

	for row in range(row_end):
		# find row at or below current one with max value 
		max_row = row + np.argmax(A[row:, row])

		# swap
		A[[row, max_row]] = A[[max_row, row]]
		b[[row, max_row]] = b[[max_row, row]]

		# elimination
		next_row = row + 1
		factors = A[next_row:, row] / A[row, row]
		A[next_row:, :] -= A[row, :] * factors[:, None] 
		b[next_row:] -= factors * b[row]

	# backward
	for row in range(row_end, -1, -1):
		next_row = row + 1
		b[row] -= np.sum(A[row, next_row:] * x[next_row:])
		x[row] = b[row] / A[row, row]

	return x