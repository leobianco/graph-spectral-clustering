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

    k = int(np.max(Z_v))
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


def visualize_eigenvectors(eigvecs, Z_v):
    # Visualization
    pcaeigvecs = PCA(n_components=2).fit_transform(eigvecs.T)
    plt.scatter(pcaeigvecs[:,0], pcaeigvecs[:,1], c=Z_v)
    plt.show()
