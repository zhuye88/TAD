import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps

class IK_iNNE:
    data = None
    centroid = []

    def __init__(self, max_samples, n_estimators, random_state=None):
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit_transform(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]
        n, d = self.data.shape
        IDX = np.array([])  # column index
        V = []
        for i in range(self.n_estimators):
            subIndex = sample(range(sn), self.max_samples)
            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :]
            tt_dis = cdist(tdata, tdata)
            radius = []  # restore centroids' radius
            for r_idx in range(self.max_samples):
                r = tt_dis[r_idx]
                r[r < 0] = 0
                r = np.delete(r, r_idx)
                radius.append(np.min(r))
            self.centroids_radius.append(radius)
            nt_dis = cdist(tdata, self.data)
            centerIdx = np.argmin(nt_dis, axis=0)
            for j in range(n):
                V.append(int(nt_dis[centerIdx[j], j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.max_samples), axis=0)
        IDR = np.tile(range(n), self.n_estimators)  # row index
        # V = np.ones(self.t * n) #value
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.n_estimators * self.max_samples))
        return ndata

    def fit(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        n_samples = self.data.shape[0]

        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            subIndex = rnd.choice(n_samples, self.max_samples, replace=False)
            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :]
            tt_dis = cdist(tdata, tdata)
            radius = []  # restore centroids' radius
            for r_idx in range(self.max_samples):
                r = tt_dis[r_idx]
                r[r < 0] = 0
                r = np.delete(r, r_idx)
                radius.append(np.min(r))
            self.centroids_radius.append(radius)
        return self

    def transform(self, newdata):
        assert self.centroid != None, "invoke fit() first!"
        n, d = newdata.shape
        IDX = np.array([])
        V = []
        for i in range(self.n_estimators):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            dis = cdist(tdata, newdata)
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                V.append(int(dis[centerIdx[j], j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.max_samples), axis=0)
        IDR = np.tile(range(n), self.n_estimators)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.n_estimators * self.max_samples))
        return ndata.toarray()
