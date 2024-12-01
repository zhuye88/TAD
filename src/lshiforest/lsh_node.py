#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

class LSHNode:
	def __init__(self, data_size=0, children={}, children_count={}, hash_func_index=-1, lof=0):
		#self._data = data
		self._data_size = data_size
		self._children = children
		self._children_count = children_count
		self._hash_func_index = hash_func_index
		self._lof = lof


	def display(self):
		print(self._hash_func_index)
		#print self._data
		print(self._data_size)
		print(self._children)
		print(self._children_count)
		print(self._lof)

	
	def get_children(self):
		return self._children

	
	def get_data_size(self):
		return self._data_size 


	def get_hash_func_index(self):
		return self._hash_func_index

		
	def get_children_count(self):
		return self._children_count


	def get_lof(self):
		return self._lof

