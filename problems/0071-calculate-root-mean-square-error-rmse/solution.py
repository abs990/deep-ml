
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	def validity_check(arr):
		return arr.size != 0 and np.issubdtype(arr.dtype, np.number)

	if validity_check(y_true) and validity_check(y_pred) and y_true.shape == y_pred.shape:
		return round(np.sqrt(np.mean((y_true - y_pred) ** 2)),3)
