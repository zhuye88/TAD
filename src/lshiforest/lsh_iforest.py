#! /usr/bin/python

"""
Created on Wed Apr 29 22:37:01 2020

@author: mq20197379
"""

from .lsh_forest import LSHForest
from .sampling import VSSampling
from .lsh import E2LSH
from .klsh import KernelLSH
from .lsh import AngleLSH

class LSHiForest(LSHForest):
	   
	"""
    LSHiForest Anomaly Detection Algorithm.
    Return the anomaly score of each sample. The lower, the more abnormal.
	LSHiForest shares a similar idea adopted by iForest, but more generic with the capability of handling different types of distance metrics underlying the definition of anomalies. The versatality of this algorithm stems from the integration of the forest isolation mechanism with locality-sensitive hashing (LSH).
    
    Parameters
    ----------
	lsh_family : string, optional (default="L2SH") 
		The possible values include 'ALSH' for angular distance, 'L1SH' for L1 (Mahattan) distance, 'L2SH' for L2 (Euclidean) distance, and 'KLSH' for the kernelized angular distance. The default value is 'L2SH' given this distance metric is commonly-used in real applications.
    num_trees : int, optional (default=100)
        The number of base estimators in the ensemble.
    granularity : int, optional (default=1)
        This parameter is to control the sensitivity of the algorithm with respect to duplicated or very similar data instances which can lead to only-one-partition case for LSH and are hard to be partitioned. If the value is '1', the model takes the lenghth of single branches of an isolation. Otherwise, the isolation will be 'virtually' compressed by just counting binary/multi-fork branches. 
            
    References
    ----------
    .. [1] Xuyun Zhang, Wanchun Dou, Qiang He, Rui Zhou, Christopher Leckie, Ramamohanarao Kotagiri, Zoran Salcic, "LSHiForest: A generic framework for fast tree isolation based ensemble anomaly analysis", IEEE Internatonal Conference on Data Engineering (ICDE), 2017.  
           
    Examples
    --------
    >>> from detectors import LSHiForest
    >>> X = [[-1.1], [0.3], [0.5], [100]]
    >>> clf = LSHiForest.fit(X)
    >>> clf.decision_function([[0.1], [0], [90]])
	>>> array([-0.21098136, -0.23885756, -0.71920724])
    """
	
	def __init__(self, lsh_family='L2SH', num_trees=100, granularity=1):
		if lsh_family == 'ALSH':
			LSHForest.__init__(self, num_trees, VSSampling(num_trees), AngleLSH(), granularity=1)
		elif lsh_family == 'L1SH':
			LSHForest.__init__(self, num_trees, VSSampling(num_trees), E2LSH(norm=1), granularity=1)
		elif lsh_family == 'KLSH':
			LSHForest.__init__(self, num_trees, VSSampling(num_trees), KernelLSH(), granularity=1)
		else:
			LSHForest.__init__(self, num_trees, VSSampling(num_trees), E2LSH(norm=2), granularity=1)