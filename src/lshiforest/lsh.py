#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np


class LSH():
	''' The base class of LSH families '''
	def __init__(self, default_pool_size=50):
		self._default_pool_size = default_pool_size

	# Virtual methods
	# type <- get_lsh_type(self) 
	# display_hash_func_parameters(self)
	# x' <- format_for_lsh(self, x)
	# key <- get_hash_value(self, x, hash_index)


class E2LSH(LSH):
	''' Class to build E2 locality sensitive hashing family '''

	def __init__(self, bin_width=4, norm=2, default_pool_size=50):
		LSH.__init__(self, default_pool_size)
		self._dimensions = -1
		self._bin_width = bin_width
		self._norm = norm
		self.A_array = None
		self.B_array = None 


	def get_lsh_type(self):
		return 'L'+str(self._norm)+'LSH'


	def display_hash_func_parameters(self):
		for i in range(len(self.A_array)):
			print (self.A_array[i], self.B_array[i])

	def fit(self, data):
		if (data == None).all():
			return
		self._dimensions = len(data[0])-1

		self.A_array = []
		self.B_array = [] 
		if self._norm == 1:
			self.A_array.append(np.random.standard_cauchy(self._dimensions))
		elif self._norm == 2:
			self.A_array.append(np.random.normal(0.0, 1.0, self._dimensions))
		self.B_array.append(np.random.uniform(0.0, self._bin_width))
		for i in range(1, self._default_pool_size):
			repeated = True
			while repeated == True:
				repeated = False
				a=[]
				if self._norm == 1:
					a=np.random.standard_cauchy(self._dimensions)
				elif self._norm == 2:
					a=np.random.normal(0.0, 1.0, self._dimensions)
				b = np.random.uniform(0, self._bin_width)
				for j in range(0, len(self.A_array)):
					if np.array_equal(a, self.A_array[j]) and b == self.B_array[j]:
						repeated = True
						break
				if repeated == False:	
					self.A_array.append(a)
					self.B_array.append(b)	


	def format_for_lsh(self, x):
		return x


	def get_hash_value(self, x, hash_index):
		cur_len = len(self.A_array)
		while hash_index >= cur_len:
			repeated = True
			while repeated == True:
				repeated = False
				a=[]
				if self._norm == 1:
					a=np.random.standard_cauchy(self._dimensions)
				elif self._norm == 2:
					a=np.random.normal(0.0, 1.0, self._dimensions)
				b = np.random.uniform(0, self._bin_width)
				for j in range(0, cur_len):
					if np.array_equal(a, self.A_array[j]) and b == self.B_array[j]:
						repeated = True
						break
				if repeated == False:
					self.A_array.append(a)
					self.B_array.append(b)
					cur_len += 1
		return int(np.floor((np.dot(x, self.A_array[hash_index])+self.B_array[hash_index])/self._bin_width))
		

class AngleLSH(LSH):
	def __init__(self, default_pool_size=50):
		LSH.__init__(self, default_pool_size)
		self._weights = None

	def get_lsh_type(self):
		return 'AngleLSH'

	def display_hash_func_parameters(self):
		for i in range(len(self._weights)):
			print(self._weights[i])

	def fit(self, data):
		if data is None:
			return
		self._dimensions = len(data[0])-1

		self._weights=[]
		# Both distributions should be ok given that they are symmetric w.r.t. 0.
                #self._weights.append(np.random.uniform(-1.0, 1.0, self._dimensions))
		self._weights.append(np.random.normal(0.0, 1.0, self._dimensions))
		for i in range(1, self._default_pool_size):
			repeated = True
			while repeated == True:
				repeated = False
				weight=np.random.normal(0.0, 1.0, self._dimensions)
				for j in range(0, len(self._weights)):
					if np.array_equal(weight, self._weights[j]):
						repeated = True
						break
				if repeated == False:	
					self._weights.append(weight)


	def format_for_lsh(self, x):
		return x

	def get_hash_value(self, x, hash_index):
		cur_len = len(self._weights)
		while hash_index >= cur_len:
			repeated = True
			while repeated == True:
				repeated = False
				weight=np.random.normal(0.0, 1.0, self._dimensions)
				for j in range(0, cur_len):
					if np.array_equal(weight, self._weights[j]):
						repeated = True
						break
				if repeated == False:	
					self._weights.append(weight)
					cur_len += 1

		return -1 if np.dot(x, self._weights[hash_index]) <0 else 1


class MinLSH(LSH):
	''' Class to build LSH for set similarity'''	
	def __init__(self, n_dimensions, default_pool_size=50):
		LSH.__init__(self, default_pool_size)
		self._num_dimensions = n_dimensions
		self._mod_base = self._get_prime_not_less_than(n_dimensions)
		self._hash_parameters_pool = np.c_[np.random.randint(0, np.iinfo('i').max, default_pool_size), np.random.randint(0, np.iinfo('i').max, default_pool_size), np.random.randint(0, np.iinfo('i').max, default_pool_size)]

	def get_lsh_type(self):
		return 'L2LSH'

	def display_hash_func_parameters(self):
		for i in range(len(self._hash_parameters_pool)):
			print(self._hash_parameters_pool[i])

	def format_for_lsh(self, x):
		return x

	def get_hash_value(self, x, hash_index):
		cur_len = len(self._hash_parameters_pool)
		while hash_index >= cur_len:
			repeated = True
			while repeated == True:
				repeated = False
				new_para = np.random.randint(0, np.iinfo('i').max, 3)
				for j in range(0, cur_len):
					if np.array_equal(new_para, self._hash_parameters_pool[j]):
						repeated = True
						break
				if repeated == False:
					self._hash_parameters_pool = np.append(self._hash_parameters_pool, [new_para], axis = 0)
					cur_len += 1

		result = np.iinfo('i').max
		for i in x:
			hash_value = self._permutationHash(i, self._hash_parameters_pool[hash_index][0], self._hash_parameters_pool[hash_index][1], self._hash_parameters_pool[hash_index][2])
			if hash_value < result:
				result = hash_value
		return result


	def _permutationHash(self, index_value, a, b, c):
		hash_value = (a*index_value+b) % self._mod_base
		return abs(hash_value)%self._num_dimensions

	def _get_prime_not_less_than(self, x):
		y = x
		if y%2 == 0 and y != 2:
			y += 1
		while not self._is_prime(y):
			y += 2
		return y

	def _is_prime(self, x):
		if x < 2:
			print('Input Error!!!')
		if x == 2 or x == 3:
			return True
		if x%2 == 0:
			return False
		if x%3 == 0:
			return False
		upper_limit = int(np.ceil(np.sqrt(x)))
		i = 1
		divisor = 6*i-1
		while divisor <= upper_limit:
			if x%divisor == 0:
				return False
			divisor += 2
			if x% divisor == 0:
				return False
			i += 1
			divisor = 6*i-1
		return True
