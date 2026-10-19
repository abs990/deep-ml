
import numpy as np

def jaccard_index(y_true, y_pred):
	num = np.sum((y_true == 1) & (y_pred == 1))
	den = np.sum((y_true == 1) | (y_pred == 1))
	if den > 0:
		result = 1.0 * num / den
	else:
		result = 0.0
	return round(result, 3)
