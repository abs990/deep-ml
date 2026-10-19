import numpy as np
def translate_object(points, tx, ty):
	
	points = np.array(points)
	points = np.hstack((points, np.ones([points.shape[0],1], points.dtype)))

	translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

	translated_points = (translate @ points.T).T
	
	return translated_points[:, :-1]
