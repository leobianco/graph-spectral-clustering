import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, eigh
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils import *


class SpectralClustering():
    """Simple wrapper of sklearn KMeans to perform spectral clustering.
    
    Args:
        k (int): number of clusters.
    """

    def __init__(self, k):
        self.k = k
    
    def cluster(self, A, sym=False):
        """Clusters the eigenvectors of the Laplacian matrix.

        Args:
            A ((n, n) np.array): adjacency matrix.
            sym (bool): whether to use symmetric (normalized) Laplacian or not.

        Returns:
            kmeans.labels_ ((n,) np.array): labels assigned to eigenvectors.
            cluster_centers_ ((n, n) np.array): centroids.
        """

        n, _ = A.shape
        D = np.diag(np.sum(A, axis=1))
        L = np.linalg.inv(sqrtm(D))@A@np.linalg.inv(sqrtm(D)) if sym else D-A
        eigvals, eigvecs = eigh(L)
        eigvecs /= np.linalg.norm(eigvecs, axis=0)
        eigvecs = eigvecs[:, (n-self.k):]  # largest k eigenvectors
        print(eigvecs)
        visualize_eigenvectors(eigvecs)
        kmeans = KMeans(n_clusters = self.k).fit(eigvecs)

        return kmeans.predict(eigvecs), kmeans.cluster_centers_.T
