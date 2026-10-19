import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	M = len(dense_matrix)
	N = len(dense_matrix[0])

	values = []
	col_indices = []
	row_pointer_arr = [0] * (M + 1)

	for i in range(M):
		row_pointer_arr[i + 1] = row_pointer_arr[i]
		for j in range(N):
			if dense_matrix[i][j] > 0:
				values.append(dense_matrix[i][j])
				col_indices.append(j)
				row_pointer_arr[i + 1] += 1

	return values, col_indices, row_pointer_arr