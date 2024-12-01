import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted
from ._ikgad import IKGAD


class IKAT(OutlierMixin, BaseEstimator):
    """Isolation-based anomaly detection using nearest-neighbor ensembles.
    The INNE algorithm uses the nearest neighbour ensemble to isolate anomalies.
    It partitions the data space into regions using a subsample and determines an
    isolation score for each region. As each region adapts to local distribution,
    the calculated isolation score is a local measure that is relative to the local
    neighbourhood, enabling it to detect both global and local anomalies. INNE has
    linear time complexity to efficiently handle large and high-dimensional datasets
    with complex distributions.
    Parameters
    ----------
    n_estimators_1 : int, default=200
        The number of base estimators in the ensemble of first step.
    max_samples_1 : int, default="auto"
        The number of samples to draw from X to train each base estimator in the first step.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples` * X.shape[0]` samples.
            - If "auto", then `max_samples=min(8, n_samples)`.
    n_estimators_2 : int, default=200
        The number of base estimators in the ensemble of secound step.
    max_samples_2 : int, default="auto"
        The number of samples to draw from X to train each base estimator in the secound step.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples` * X.shape[0]` samples.
            - If "auto", then `max_samples=min(8, n_samples)`.
    method: {"inne", "anne", "auto"}, default="inne"
        isolation method to use. The original algorithm in paper is `"inne"`.
    contamination : "auto" or float, default="auto"
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.
            - If "auto", the threshold is determined as in the original paper.
            - If float, the contamination should be in the range (0, 0.5].
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    References
    ----------
    .. [1] Wang, Y., Wang, Z., Ting, K. M., & Shang, Y. (2024).
    A Principled Distributional Approach to Trajectory Similarity Measurement and
    its Application to Anomaly Detection. Journal of Artificial Intelligence Research, 79, 865-893.
    --------
    >>> from isoml.trajectory.anomaly import IKAT
    >>> from isoml.trajectory.datasets import SheepDogs
    >>> sheepdogs = SheepDogs()
    >>> X, y = sheepdogs.load(return_X_y=True)
    >>> clf = IKAT().fit(X)
    >>> clf.predict(X)
    array([ 1,  1, -1])
    """

    def __init__(
        self,
        n_estimators_1=200,
        max_samples_1="auto",
        n_estimators_2=200,
        max_samples_2="auto",
        contamination="auto",
        method="inne",
        random_state=None,
    ):
        self.n_estimators_1 = n_estimators_1
        self.max_samples_1 = max_samples_1
        self.n_estimators_2 = n_estimators_2
        self.max_samples_2 = max_samples_2
        self.random_state = random_state
        self.contamination = contamination
        self.method = method

    def fit(self, X, y=None):
        """
        Fit estimator.
        Parameters
        ----------
        X : 3D array-like of shape (n_trajectories , n_points, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """

        # TODO: Check 3D data
        # X = check_array(X, accept_sparse=False)
        n_samples = len(X)

        self._fit(X)
        self.is_fitted_ = True

        if self.contamination != "auto":
            if not (0.0 < self.contamination <= 0.5):
                raise ValueError(
                    "contamination must be in (0, 0.5], got: %f" % self.contamination
                )

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # we take the opposite as we consider the opposite of their score.
            self.offset_ = -0.5
        else:
            # else, define offset_ wrt contamination parameter
            self.offset_ = np.percentile(
                self.score_samples(X), 100.0 * self.contamination
            )

        return self

    def _fit(self, X):
        ikgod = IKGAD(
            n_estimators_1=self.n_estimators_1,
            max_samples_1=self.max_samples_1,
            n_estimators_2=self.n_estimators_2,
            max_samples_2=self.max_samples_2,
            random_state=self.random_state,
            contamination=self.contamination,
            method=self.method,
        )
        self.ikgod = ikgod.fit(X)
        return self

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
        """

        check_is_fitted(self)

        return self.ikgod.predict(X)

    def decision_function(self, X):
        """
        Average anomaly score of X of the base classifiers.
        The anomaly score of an input sample is computed as
        the mean anomaly score of the .
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.
        """
        # We subtract self.offset_ to make 0 be the threshold value for being
        # an outlier.

        return self.ikgod.decision_function(X)

    def score_samples(self, X):
        """
        Opposite of the anomaly score defined in the original paper.
        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.
        """

        check_is_fitted(self, "is_fitted_")
        # TODO: Check 3D data
        # X = check_array(X, accept_sparse=False)

        scores = self.ikgod.score_samples(X)

        return scores
