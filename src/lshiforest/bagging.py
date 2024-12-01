#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np

class Bagging:

	def __init__(self):
		self._num_dimensions = -1
		self._rotated_axis_system = None

	
	def fit(self, data):

		if data == None:
			return

		data = np.array(data)
		data_size = len(data)
		original_dim = len(data[0])-1

		if original_dim < 4:
			self._num_dimensions = original_dim
			self._rotated_axis_system = np.identity(original_dim) 
			return 

		self._num_dimensions = 2+int(np.ceil(0.5*np.sqrt(original_dim)))

		orthonormal_matrix = []
		for i in range(original_dim):
			orthonormal_matrix.append(np.random.uniform(-1.0, 1.0, self._num_dimensions))
		orthonormal_matrix = np.array(orthonormal_matrix)
		self._rotated_axis_system = np.linalg.qr(orthonormal_matrix)[0]


	def get_transformed_data(self, data):
		indices = data[:, 0]
		values = data[:, 1:]
		return np.c_[indices, np.dot(values, self._rotated_axis_system)]
		
