#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np
import copy as cp

class Sampling():

	def __init__(self, num_samples, bootstrap=False, bagging=None):
		self._num_samples = num_samples
		self._bootstrap = bootstrap
		self._bagging = bagging
		self._bagging_instances = None


class VSSampling(Sampling):
	
	def __init__(self, num_samples, lower_bound=64, upper_bound=1024, vs_type='EXP', bootstrap=False, bagging=None):
		Sampling.__init__(self, num_samples, bootstrap, bagging)
		self._lower_bound = lower_bound
		self._upper_bound = upper_bound
		self._vs_type = vs_type
    

	def fit(self, data):
		if data is None:
			return
		if self._bagging != None:
			self._bagging_instances = []
			for i in range(self._num_samples):
				self._bagging.fit(data)
				self._bagging_instances.append(cp.deepcopy(self._bagging))


	def draw_samples(self, data):
		
		data_size = len(data)
		
		sampled_indices = []
		if self._bootstrap:
			for i in range(self._num_samples):
				indices = []
				if self._vs_type == 'FIX':
					indices = np.random.randint(0, data_size, self._lower_bound)
				elif self._vs_type == 'UNI':
					indices = np.random.randint(0, data_size, int(round(np.random.uniform(self._lower_bound, self._upper_bound))))
				elif self._vs_type == 'EXP':
					indices = np.random.randint(0, data_size, int(round(np.power(2.0, np.random.uniform(np.log2(self._lower_bound), np.log2(self._upper_bound))))))
					
				indices.sort()
				sampled_indices.append(indices)
		else:
			for i in range(self._num_samples):
				sample_size = 0
				if self._vs_type == 'FIX':
					sample_size = min(data_size, self._lower_bound)
				if self._vs_type == 'UNI':
					sample_size = min(data_size, int(round(np.random.uniform(self._lower_bound, self._upper_bound))))
				elif self._vs_type == 'EXP':
					sample_size = min(data_size, int(round(np.power(2.0, np.random.uniform(np.log2(self._lower_bound), np.log2(self._upper_bound))))))

                #remain_indices = range(data_size)
				remain_indices = list(range(data_size))
				new_indices = []
				for j in range(sample_size):
					index_ind = np.random.randint(0, len(remain_indices), 1)[0]
					index_val = remain_indices[index_ind]
					new_indices.append(index_val)
					remain_indices.remove(index_val)
				new_indices.sort()
				sampled_indices.append(new_indices)

		sampled_datas = []
		for i in range(self._num_samples):
			transformed_data = data
			if self._bagging != None:
				transformed_data = self._bagging_instances[i].get_transformed_data(data)
			sampled_data = []
			for j in sampled_indices[i]:
				sampled_data.append(transformed_data[j])
			sampled_datas.append(sampled_data)
				
		return sampled_datas
