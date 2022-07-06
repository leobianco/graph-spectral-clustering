import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from sklearn.decomposition import PCA


def accuracy(labels, Z_v):
    """Calculates the classification accuracy under label permutations.

    Args:
        labels ((n,) np.array): vector with k-means estimated community labels.
        Z_v ((n, k) np.array): ground truth vector of community labels.

    Returns:
        accuracy (float): maximal percentage of correct class predictions.
        saved_permutation (k np.array): ``correct'' permutation of labels.
    """

    k = np.unique(Z_v).shape[0]
    accuracy = 0
    all_permutations = list(itertools.permutations(list(range(k))))
    saved_permutation = None
    for permutation in all_permutations:
        labels_permuted = np.select([labels==i for i in range(k)], permutation)
        curr_accuracy = np.mean(labels_permuted==Z_v)
        if curr_accuracy > accuracy:
            accuracy = curr_accuracy
            saved_permutation = permutation

    return accuracy, saved_permutation


def visualize_eigenvectors(eigvecs, centroids):
    n_centroids = centroids.shape[0]
    pca = PCA(n_components=2).fit_transform(np.vstack((eigvecs, centroids)))
    plt.scatter(pca[:-n_centroids,0], pca[:-n_centroids,1])
    plt.scatter(pca[-n_centroids:,0], pca[-n_centroids:, 1], c='orange')
    plt.show()


def visualize_eigenvalues(eigvals):
    plt.plot(eigvals)
    plt.scatter(list(range(eigvals.shape[0])), eigvals, c='darkblue')
    plt.show()


def draw_graph(A, Z_v):
    """Wrapper for networkx drawing capabilities.

    Args:
        A ((n, n) np.array): adjacency matrix of the graph to be shown.
        permutation ((k,) np.array): permutation of labels maximizing accuracy.
        Z_v (n np.array): vector with true community labels.
        tau ((n, k) np.array): matrix of variational parameters.

    Returns:
        None (pyplot window with graph)
    """

    n = A.shape[0]
    G = nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, node_color=Z_v, with_labels=False)
    plt.show();
