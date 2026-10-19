import numpy as np
def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	# Your code here
	y_true = y_true if isinstance(y_true, list) else [y_true]
	y_pred = y_pred if isinstance(y_pred, list) else [y_pred]
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)

	y_abs_diff = np.abs(y_true - y_pred)
	mask = y_abs_diff <= delta

	mse_idx = np.where(mask)[0]
	mae_idx = np.where(~mask)[0]

	loss = np.zeros_like(y_true)
	if mse_idx.size != 0:
		loss[mse_idx] = 0.5 * (y_abs_diff[mse_idx] ** 2)
	if mae_idx.size != 0:
		loss[mae_idx] = delta * (y_abs_diff[mae_idx] - 0.5 * delta)

	return np.mean(loss)