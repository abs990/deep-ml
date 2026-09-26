import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	
	# infer number of categories if needed
	if n_col is None:
		n_col  = 1 + max(x)

	# to store result
	categorical = np.zeros((x.shape[0], n_col))

	# one hot encoding
	categorical[np.arange(categorical.shape[0]), x] = 1

	return categorical