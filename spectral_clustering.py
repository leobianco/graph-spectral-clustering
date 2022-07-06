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
    
    def cluster(self, A, unn=False, largest=False):
        """Clusters the eigenvectors of the Laplacian matrix.

        Args:
            A ((n, n) np.array): adjacency matrix.
            unn (bool): whether to use unnormalized Laplacian or not.
            largest (bool): whether to use largest (Rohe) or smallest (von
            Luxburg) k eigenvectors of the Laplacian.

        Returns:
            kmeans.labels_ ((n,) np.array): labels assigned to eigenvectors.
            cluster_centers_ ((n, n) np.array): centroids.
        """

        n, _ = A.shape

        # Observed
        D = np.diag(np.sum(A, axis=1))
        L = D-A if unn\
                else np.eye(n)-np.linalg.inv(sqrtm(D))@A@np.linalg.inv(sqrtm(D))
        eigvals, eigvecs = eigh(L)
        eigvecs = eigvecs[:, (n-self.k):] if largest else eigvecs[:, :self.k]
        eigvecs /= np.linalg.norm(eigvecs, axis=0)
        visualize_eigenvectors(eigvecs)
        kmeans = KMeans(n_clusters = self.k).fit(eigvecs)

        return kmeans.predict(eigvecs), kmeans.cluster_centers_.T
