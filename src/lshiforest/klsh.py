#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np
from sklearn.metrics import pairwise, pairwise_kernels
#from scipy.misc import comb
from scipy.special import comb
import types as tp

from .lsh import LSH


class KernelLSH(LSH):
	''' Class to build kernelized locality sensitive hashing function '''

	def __init__(self, nbits=3, kernel='rbf', kernel_kwds='range', para_p_max=300, para_p_exp=0.5, para_t_max=30, para_t_ratio=4, weight_pool_size=50):
		LSH.__init__(self, weight_pool_size)
		self._nbits = nbits
		self._kernel_func = None
		self._kernel_kwds = kernel_kwds
		self._check_kernel(kernel, kernel_kwds)
		self._train_data = None
		self._para_p = para_p_max
		self._para_p_exp = para_p_exp
		self._para_t = para_t_max
		self._para_t_ratio = para_t_ratio
		self._K_half = None
		self._weight_pool_size = weight_pool_size
		self._e_s_pool = None 
		self._weight_pool = None 
		self._k = None


	def get_lsh_type(self):
		return 'KLSH'

	
	def display_hash_func_parameters(self):
		print (self._e_s_pool)
		print (self._weight_pool)
		return None


	def format_for_lsh(self, data):
		return np.c_[data[:, 0], self.compute_test_kernel(data[:, 1:]).T.A]


	def _check_kernel(self, kernel, kernel_kwds=None):
		self._kernel_func = kernel


	def fit(self, data):

		data = np.array(data)	
		data_size = len(data)
		
		train_data_size = min(int(np.ceil(pow(data_size, self._para_p_exp))), self._para_p)
		train_data = data[np.random.choice(data_size, train_data_size, False)][:, 1:]

		X = self._train_data = np.mat(train_data)
		self._para_p, d = X.shape
		self._para_t = min(max(1, self._para_p//self._para_t_ratio), self._para_t)
		#print self._para_p, self._para_t
		
		if self._kernel_func == 'rbf':
#			if type(self._kernel_kwds) == tp.StringType and self._kernel_kwds == 'auto':
			if type(self._kernel_kwds) == str and self._kernel_kwds == 'auto':
				self._kernel_kwds = 1.0/d
#			elif type(self._kernel_kwds) == tp.StringType and self._kernel_kwds == 'range':
			elif type(self._kernel_kwds) == str and self._kernel_kwds == 'range':
				self._kernel_kwds = pow(10, np.random.uniform(np.log10(0.1/d), np.log10(1.0/d)))

		# Compute the kernel matrix
		K = self._compute_kernel(X, X)
		self._K = K.copy()
		
		# Center the kernel matrix
		KMean0 = K.mean(0)
		KMean1 = K.mean(1)
		KMean = K.mean()
		K -= KMean0
		K -= KMean1
		K += KMean

		# Compute K^-0.5 via the eigendecomposition
		E, V = np.linalg.eigh(K)
		E[E>1e-8] = E[E>1e-8]**-0.5
		E[E<=1e-8] = 0
		E = np.diag(E)
		self._K_half = V*E*V.T

		# Construct default weight pool
		self._e_s_pool = []
		self._weight_pool = []
		self._extend_weight_pool(self._weight_pool_size)


	def get_train_kernel(self):
		return self._K


	def _compute_kernel(self, X, Y):
		n_X, d_X = X.shape
		n_Y, d_Y = Y.shape
		if d_X != d_Y:
			print('Dimensions not matched!!!', d_X, d_Y)
			return None	

		part_X = sum(np.power(X.T, 2)).T * np.mat(np.ones((1, n_Y)))
		part_Y = np.mat(np.ones((n_X, 1))) * sum(np.power(Y.T, 2))
		part_XY = -2 * X * Y.T
		squared_diff = part_X+part_Y+part_XY

		if self._kernel_func == 'rbf':
			return np.exp(-1.0*self._kernel_kwds*(squared_diff))

		if self._kernel_func == 'puk':
			gamma = 1.0/d_X
			omega = self._kernel_kwds
			return np.power((1.0+8*gamma*(np.power(2.0, 1.0/omega)-1.0)*squared_diff), -omega) 


	def _extend_weight_pool(self, extended_size):
		# There is a upper-bound for the number of combinations
		size = min(extended_size, int(comb(self._para_p, self._para_t))**self._nbits-len(self._e_s_pool))

		if size > 0 and len(self._e_s_pool) == 0:
			new_e_s = self._generate_e_s()
			self._e_s_pool.append(new_e_s)	
			self._weight_pool.append(np.sqrt((self._para_p-1.0)/self._para_t)*self._K_half*new_e_s)
			size -= 1 

		for i in range(size):
			repeated = True
			while repeated == True:
				repeated = False
				new_e_s = self._generate_e_s()
				for e_s in self._e_s_pool:
					if np.array_equal(new_e_s, e_s):
						repeated = True
						break 		
				if repeated == False:
					self._e_s_pool.append(new_e_s)	
					self._weight_pool.append(np.sqrt((self._para_p-1.0)/self._para_t)*self._K_half*new_e_s)
		
		self._weight_pool_size = len(self._weight_pool) 

		
	def _generate_e_s(self):
		e_s = np.zeros((self._para_p, self._nbits))
		i = np.array([np.random.choice(self._para_p, self._para_t, False) for i in range(self._nbits)]).T
		e_s[i, range(self._nbits)] = 1
		return np.mat(e_s)

	
	def compute_test_kernel(self, x):
		return self._compute_kernel(self._train_data, np.mat(x))


	def get_hash_value(self, test_kernel, hash_index):
		while hash_index >= self._weight_pool_size:
			self._extend_weight_pool(1)

		# Compute weight
		weight = self._weight_pool[hash_index] 
		test_kernel = np.mat(test_kernel).T

		bits = weight.T*test_kernel > 0	
		integer = 0
		for i in range(self._nbits):
			integer += bits[i, 0]*2**(i)
		return integer
					
		
	def get_multiple_bits(self, x, num_keys):
		keys = ''
		for i in range(num_keys):
			keys += str(self.get_hash_value(x, i))
		return keys
