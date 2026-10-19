import numpy as np
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	
	actual = np.array(actual)
	predicted = np.array(predicted)

	TP = int(np.sum((actual == 1) & (predicted == 1)))
	FN = int(np.sum((actual == 1) & (predicted == 0)))
	FP = int(np.sum((actual == 0) & (predicted == 1)))
	TN = int(np.sum((actual == 0) & (predicted == 0)))

	confusion_matrix = [[TP, FN], [FP, TN]]
	accuracy = (TP + TN) / (TP + TN + FP + FN)
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	f1 = 2 * precision * recall / (precision + recall)
	specificity = TN / (TN + FP)
	negativePredictive = TN / (TN + FN)

	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
