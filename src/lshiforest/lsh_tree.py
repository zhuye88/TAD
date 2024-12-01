#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np

from . import lsh_node as nd

class LSHTree:
	def __init__(self, lsh):
		self._lsh = lsh
		self._depth_limit = 0
		self._root = None
		self._n_samples = 0 
		self._branch_factor = 0
		self._reference_path_length = 0
		

	def build(self, data):
		self._n_samples = len(data)
		self._depth_limit = self.get_random_height(self._n_samples)
		data = np.array(data)
		data = self._lsh.format_for_lsh(data)
		self._root = self._recursive_build(data, self._depth_limit)
		self._branch_factor = self._get_avg_branch_factor()
		self._reference_path_length = self.get_random_path_length_symmetric(self._n_samples)
		


	def _recursive_build(self, data, depth_limit, lof=0, hash_func_index=0):
		n_samples = len(data)
		if n_samples == 0:
			return None
		if n_samples == 1 or hash_func_index > depth_limit:
			return nd.LSHNode(len(data), {}, {}, hash_func_index, lof)
		else:
			cur_index = hash_func_index
			partition = self._split_data(data,cur_index)
			while len(partition) == 1 and cur_index <= depth_limit:
				cur_index += 1
				partition = self._split_data(data, cur_index)
			if cur_index > depth_limit:
				return nd.LSHNode(len(data), {}, {}, cur_index, lof)

			children_count = {}
			for key in partition.keys():
				children_count[key] = len(partition.get(key))

			mean = np.mean(list(children_count.values()))
			std = np.std(list(children_count.values()))
				
			children = {}
			for key in partition.keys():
				child_data = partition.get(key)
				children[key] = self._recursive_build(child_data, depth_limit, min(0.0, (children_count[key]-mean)/std), cur_index+1)
			return nd.LSHNode(len(data), children, children_count, cur_index, lof)


	def _split_data(self, data, depth):
		''' Split the data using LSH '''
		partition = {}
		for i in range(len(data)):
			key = self._lsh.get_hash_value(np.array(data[i][1:]), depth)
			if key not in partition:
				partition[key] = [data[i]]
			else:
				sub_data = partition[key]
				sub_data.append(data[i])
				partition[key] = sub_data
		return partition


	def get_num_instances(self):
		return self._n_samples


	def display(self):
		self._recursive_display(self._root)	

	
	def _recursive_display(self, lsh_node, leftStr=''):
		if lsh_node is None:
			return
		children = lsh_node.get_children()

		print(leftStr+'('+str(len(leftStr))+','+str(lsh_node._hash_func_index)+'):'+str(lsh_node._data_size)+':'+str(lsh_node._children_count)+','+str(lsh_node._lof))
		
		for key in children.keys():
			self._recursive_display(children[key], leftStr+' ')


	def predict(self, granularity, point):
		point = self._lsh.format_for_lsh(np.mat(point)).A1
		path_length = self._recursive_get_search_depth(self._root, 0, granularity, point)
		return pow(2.0, (-1.0*path_length/self._reference_path_length))
			 

	def _recursive_get_search_depth(self, lsh_node, cur_depth, granularity, point):
		if lsh_node is None:
			return -1
		children = lsh_node.get_children()
		if not children:
			real_depth = lsh_node._hash_func_index
			adjust_factor = self.get_random_path_length_symmetric(lsh_node.get_data_size())
			return cur_depth*np.power(1.0*real_depth/max(cur_depth, 1.0), granularity)+adjust_factor
		else:
			key = self._lsh.get_hash_value(point[1:], lsh_node.get_hash_func_index())
			
			if key in children.keys():
				return self._recursive_get_search_depth(children[key], cur_depth+1, granularity, point)
			else:
				cur_depth = cur_depth+1
				real_depth = lsh_node._hash_func_index+1
				return cur_depth*np.power(1.0*real_depth/max(cur_depth, 1), granularity)


	def get_avg_branch_factor(self):
		return self._branch_factor

					
	def _get_avg_branch_factor(self):
		i_count, bf_count = self._recursive_sum_BF(self._root)
		# Single node PATRICIA trie
		if i_count == 0:
			return 2.0
		return bf_count*1.0/i_count	


	def _recursive_sum_BF(self, lsh_node):
		if lsh_node is None:
			return None, None
		children = lsh_node.get_children()
		if not children:
			return 0, 0
		else:
			i_count, bf_count = 1, len(children)
			for key in children.keys():
				i_c, bf_c = self._recursive_sum_BF(children[key])
				i_count += i_c
				bf_count += bf_c
			return i_count, bf_count	

	
	def get_random_path_length_symmetric(self, num_samples):
		if num_samples <= 1:
			return 0
		elif num_samples > 1 and num_samples <= round(self._branch_factor):
			return 1
		else:
			return (np.log(num_samples)+np.log(self._branch_factor-1.0)+0.5772)/np.log(self._branch_factor)-0.5
	
	
	def get_random_height(self, num_samples):
		return 2*np.log2(num_samples)+0.8327
