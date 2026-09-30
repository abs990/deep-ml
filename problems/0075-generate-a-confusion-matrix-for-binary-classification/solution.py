
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	# init
	confusion_matrix = [[0, 0], [0, 0]]

	# processing
	for pair in data:
		confusion_matrix[1 - pair[0]][1 - pair[1]] += 1

	return confusion_matrix