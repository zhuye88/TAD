#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np
import copy as cp

from . import lsh_tree as lt
from . import lsh
from . import klsh
from . import sampling as sp
from . import bagging as bg

class LSHForest:
	def __init__(self, num_trees, sampler, lsh_family, granularity=1):
		self._num_trees = num_trees
		self._sampler = sampler
		self._lsh_family = lsh_family
		self._granularity = granularity
		self._trees = []

	
	def display(self):
		for t in self._trees:
			t.display()


	def fit(self, data):
		self.build(data)


	def build(self, data):
		indices = range(len(data))
		# Uncomment the following code for continuous values
		data = np.c_[indices, data]

		# Important: clean the tree array
		self._trees = []

		# Sampling data
		self._sampler.fit(data)
		sampled_datas = self._sampler.draw_samples(data)
		
		# Build LSH instances based on the given data
		lsh_instances = []
		for i in range(self._num_trees):
			transformed_data = data
			if self._sampler._bagging != None:
				transformed_data = self._sampler._bagging_instances[i].get_transformed_data(data)	
			self._lsh_family.fit(transformed_data)
			lsh_instances.append(cp.deepcopy(self._lsh_family))

		# Build LSH trees
		for i in range(self._num_trees):
			sampled_data = sampled_datas[i]
			tree = lt.LSHTree(lsh_instances[i])
			tree.build(sampled_data)
			self._trees.append(tree)

	
	def decision_function(self, data):
		indices = range(len(data))
		# Uncomment the following code for continuous data
		data = np.c_[indices, data]

		depths=[]
		data_size = len(data)
		for i in range(data_size):
			d_depths = []
			for j in range(self._num_trees):
				transformed_data = data[i]
				if self._sampler._bagging != None:
					transformed_data = self._sampler._bagging_instances[j].get_transformed_data(np.mat(data[i])).A1
				d_depths.append(self._trees[j].predict(self._granularity, transformed_data))
			depths.append(d_depths)
	
		# Arithmatic mean	
		avg_depths=[]
		for i in range(data_size):
			depth_avg = 0.0
			for j in range(self._num_trees):
				depth_avg += depths[i][j]
			depth_avg /= self._num_trees
			avg_depths.append(depth_avg)

		avg_depths = np.array(avg_depths)
		return -1.0*avg_depths


	def get_avg_branch_factor(self):
		sum = 0.0
		for t in self._trees:
			sum += t.get_avg_branch_factor()
		return sum/self._num_trees		
