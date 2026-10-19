def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	M = len(dense_matrix)
	N = len(dense_matrix[0])
	
	vals = []
	row_idx = []
	col_ptr = [0] * (N + 1)

	for j in range(N):
		col_ptr[j + 1] = col_ptr[j]
		for i in range(M):
			if dense_matrix[i][j] > 0:
				vals.append(dense_matrix[i][j])
				row_idx.append(i)
				col_ptr[j + 1] += 1

	return vals, row_idx, col_ptr