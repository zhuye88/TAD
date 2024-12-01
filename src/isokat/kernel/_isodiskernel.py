"""
isoml (c) by Xin Han

isoml is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import math
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from ._isokernel import IsoKernel


class IsoDisKernel(BaseEstimator, TransformerMixin):
    """Isolation Distributional Kernel is a new way to measure the similarity between two distributions.

    It addresses two key issues of kernel mean embedding, where the kernel employed has:
    (i) a feature map with intractable dimensionality which leads to high computational cost;
    and (ii) data independency which leads to poor detection accuracy in anomaly detection.

    Parameters
    ----------
    method : str, default="anne"
        The method to compute the isolation kernel feature. The available methods are: `anne`, `inne`, and `iforest`.

    n_estimators : int, default=200
        The number of base estimators in the ensemble.

    max_samples : int, default="auto"
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples` * X.shape[0]` samples.
            - If "auto", then `max_samples=min(8, n_samples)`.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    References
    ----------
    .. [1] Kai Ming Ting, Bi-Cun Xu, Takashi Washio, and Zhi-Hua Zhou. 2020.
    "Isolation Distributional Kernel: A New Tool for Kernel based Anomaly Detection".
    In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20).
    Association for Computing Machinery, New York, NY, USA, 198-206.

    Examples
    --------
    >>> from isoml.kernel import IsoDisKernel
    >>> import numpy as np
    >>> X = [[0.4,0.3], [0.3,0.8], [0.5,0.4], [0.5,0.1]]
    >>> idk = IsoDisKernel.fit(X)
    >>> D_i = [[0.4,0.3], [0.3,0.8]]
    >>> D_j = [[0.5, 0.4], [0.5, 0.1]]
    >>> idk.similarity(D_j, D_j)
    """

    def __init__(
        self, method="anne", n_estimators=200, max_samples="auto", random_state=None
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.method = method

    def fit(self, X):
        """Fit the model on data X.
        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        Returns
        -------
        self : object
        """
        X = check_array(X)
        iso_kernel = IsoKernel(
            self.method, self.n_estimators, self.max_samples, self.random_state
        )
        self.iso_kernel_ = iso_kernel.fit(X)
        self.is_fitted_ = True
        return self

    def kernel_mean(self, X):
        """Compute the kernel mean embedding of X."""
        if sp.issparse(X):
            return np.asarray(X.mean(axis=0)).ravel()
        return np.mean(X, axis=0)

    def kme_similarity(self, kme_D_i, kme_D_j, dense_output=True, is_normalize=False):
        if is_normalize:
            return np.dot(kme_D_i, kme_D_j) / (
                math.sqrt(np.dot(kme_D_i, kme_D_i))
                * math.sqrt(np.dot(kme_D_j, kme_D_j))
            )
        return np.dot(kme_D_i, kme_D_j, dense_output=dense_output) / self.n_estimators

    def similarity(self, D_i, D_j, is_normalize=True):
        """Compute the isolation distribution kernel of D_i and D_j.
        Parameters
        ----------
        D_i: array-like of shape (n_instances, n_features)
            The input instances.
        D_j: array-like of shape (n_instances, n_features)
            The input instances.
        is_normalize: whether return the normalized similarity matrix ranged of [0,1]. Default: False
        Returns
        -------
        The Isolation distribution similarity of given two dataset.
        """
        emb_D_i, emb_D_j = self.transform(D_i, D_j)
        kme_D_i, kme_D_j = self.kernel_mean(emb_D_i), self.kernel_mean(emb_D_j)
        return self.kme_similarity(kme_D_i, kme_D_j, is_normalize=is_normalize)

    def transform(self, D_i, D_j):
        """Compute the isolation kernel feature of D_i and D_j.
        Parameters
        ----------
        D_i: array-like of shape (n_instances, n_features)
            The input instances.
        D_j: array-like of shape (n_instances, n_features)
            The input instances.
        Returns
        -------
        The finite binary features based on the kernel feature map.
        The features are organised as a n_instances by psi*t matrix.
        """
        check_is_fitted(self)
        D_i = check_array(D_i)
        D_j = check_array(D_j)
        return self.iso_kernel_.transform(D_i), self.iso_kernel_.transform(D_j)
