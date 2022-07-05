import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils import visualize_eigenvectors


class SpectralClustering():
    """Simple wrapper of sklearn KMeans to perform spectral clustering.
    
    Args:
        k (int): number of clusters.
    """

    def __init__(self, k):
        self.k = k
    
    def cluster(self, A, Z_v=None):
        """Clusters the eigenvectors of the Laplacian L = D - A.

        Args:
            A ((n, n) np.array): adjacency matrix.
            Z_v ((n,) np.array): true labels array. Added for visualization
            purposes.

        Returns:
            kmeans.labels_ ((n,) np.array): labels assigned to eigenvectors.
            cluster_centers_ ((n, n) np.array): centroids.
        """

        D = np.diag(np.sum(A, axis=1))
        L = np.linalg.inv(sqrtm(D)) @ A @ np.linalg.inv(sqrtm(D))
        eigvecs = np.linalg.eigh(L)[1]
        eigvecs /= np.linalg.norm(eigvecs, axis=0)
        if Z_v is not None:
            visualize_eigenvectors(eigvecs, Z_v)
        kmeans = KMeans(n_clusters = self.k).fit(eigvecs.T)
        return kmeans.labels_, kmeans.cluster_centers_.T
