import numpy as np
import scipy.sparse as sp
from isoml.kernel import IsoGraphKernel
from sknetwork.data import art_philo_science


def test_IsoGraphKernel_fit_transform():
    graph = art_philo_science(metadata=True)
    adjacency = graph.adjacency
    features = graph.biadjacency
    names = graph.names
    names_features = graph.names_col
    names_labels = graph.names_labels
    labels_true = graph.labels
    position = graph.position

    features = features.toarray()

    # Create an instance of IsoGraphKernel
    igk = IsoGraphKernel(max_samples=8, method="anne")
    weights = [1] * len(labels_true)

    # Fit and transform the data
    transformed_data = igk.fit_transform(adjacency=adjacency, features=features, h=5)

    # Check the shape of the transformed data


#  assert transformed_data.shape == (3, 2)


# def test_IsoGraphKernel_similarity():
#     # Create a sample adjacency matrix
#     adjacency = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
#
#     # Create a sample feature matrix
#     features = np.array([[0.4, 0.3], [0.3, 0.8], [0.5, 0.4]])
#
#     # Create an instance of IsoGraphKernel
#     igk = IsoGraphKernel()
#
#     # Fit the data
#     igk.fit(adjacency)
#
#     # Compute the similarity matrix
#     similarity_matrix = igk.similarity(features)
#
#     # Check the shape of the similarity matrix
#     assert similarity_matrix.shape == (3, 3)
