
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	y_true_mean = np.mean(y_true)
	ratio = 1.0 * np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - y_true_mean) ** 2))
	return round(1 - ratio, 3)