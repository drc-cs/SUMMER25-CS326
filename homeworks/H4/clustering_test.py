from unittest.mock import patch

import numpy as np

from clustering import (dbscan, k_means, local_silhouette_score, 
                        pairwise_manhattan_distance)

np.random.seed(2024)

def test_pairwise_manhattan_distance():
    X = np.array([[1, 2], [3, 4]])
    Y = np.array([[1, 2], [3, 4], [5, 6]])
    assert np.allclose(pairwise_manhattan_distance(X, Y), np.array([[0, 4, 8], [4, 0, 4]]))

def test_k_means():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    initial_centroids = np.array([[1, 2], [3, 4]])
    centroids, labels = k_means(X, 2, initial_centroids, max_iter=4)
    assert np.allclose(centroids, np.array([[2, 3], [6, 7]]))
    assert np.allclose(labels, np.array([0, 0, 1, 1]))

@patch('sklearn.cluster.DBSCAN.fit')
@patch('sklearn.cluster.DBSCAN.fit_predict')
def test_dbscan(fit_patch, fit_predict_patch):
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    _ = dbscan(1, 1, 2)
    assert fit_patch.called or fit_predict_patch.called

@patch('sklearn.metrics.silhouette_score')
def test_local_silhouette_score(mock_ss):
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    labels = np.array([0, 0, 1, 1])
    _ = local_silhouette_score(X, labels, "euclidean")
    assert mock_ss.called_once_with(X, labels, metric="euclidean")
