import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	# counts
	tp = np.sum((y_true == 1) & (y_pred == 1))
	fp = np.sum((y_true == 0) & (y_pred == 1))
	fn = np.sum((y_true == 1) & (y_pred == 0))
	
	precision = 0.0
	if tp > 0 or fp > 0:
		precision = 1.0 * tp / (tp + fp)
	
	recall = 0.0
	if tp > 0 or fn > 0:
		recall = 1.0 * tp / (tp + fn)

	f_score = 0.0
	beta_sq = beta ** 2
	den = beta_sq * precision + recall
	if den > 0:
		f_score = (1.0 + beta_sq) * precision * recall / den

	return round(f_score, 3)