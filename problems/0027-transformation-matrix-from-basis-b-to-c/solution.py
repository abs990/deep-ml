import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	return np.matmul(np.linalg.inv(np.array(C)), np.array(B)).tolist()