import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	m, n = new_shape

	# sanity check
	total_count = 0
	for row in a:
		total_count += len(row)
	
	if total_count != m * n:
		return []

	# reshape
	return np.array(a).reshape((m, n)).tolist()