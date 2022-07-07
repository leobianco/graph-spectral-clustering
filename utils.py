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


def visualize_eigenvectors(eigvecs, centroids, labels, permutation):

    k = centroids.shape[0]  # which is also the number of centroids
    pca = PCA(n_components=2).fit_transform(np.vstack((eigvecs, centroids)))

    # Build correct labels for centroids (must permute).
    #centroid_label = np.zeros(k)
    #for i in range(k):
    #    idx = np.argwhere(labels==i)[0]
    #    representant = eigvecs[idx,:]
    #    distances =\
    #           [np.linalg.norm(representant - centroids[j,:]) for j in range(k)]
    #    centroid = np.argmin(distances)
    #    labels_permuted = np.select([labels==i for i in range(k)], permutation)
    #    centroid_label[centroid] = labels_permuted[idx]

    # pca[:-k, :] = (pca[-k:, :])[centroid_label,:]  # correct labels
    plt.scatter(pca[:-k,0], pca[:-k,1])
    plt.scatter(pca[-k:,0], pca[-k:, 1], c='orange')
    for i in range(k):
        #plt.annotate(f'{int(centroid_label[i]) + 1}', (pca[-(i+1),0], pca[-(i+1), 1]),
        #        size=15, xytext=(10, 10), textcoords='offset points')
        plt.annotate(f'{i+1}', (pca[-(i+1),0], pca[-(i+1), 1]),
                size=15, xytext=(10, 10), textcoords='offset points')
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


def load_graph(file_name):
    """Loads saved graphs

    Args:
        file_name (string): name of file to be loaded.

    Returns:
        Parameters and sample information.
    """

    with open('saved_graphs/'+file_name+'.npz', 'rb') as f:
        container = np.load(f)
        Gamma, Pi, Z, Z_v, A = [container[key] for key in container]

    return Gamma, Pi, Z, Z_v, A
