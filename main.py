import numpy as np
import argparse
from sbm import SBM
from spectral_clustering import SpectralClustering
from sklearn.cluster import SpectralClustering as skSpectralClustering
from utils import *
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument(
        '-k', help='Number of communities',
        type=int, default=2
        )
parser.add_argument(
        '-n', help='Number of points in graph to be generated',
        type=int, default=100
        )
parser.add_argument(
        '-vis', '--visual',
        help='Whether to visualize generated graph or not.', action='store_true'
        )
parser.add_argument(
        '-unn', '--unnormalized',
        help='Whether to use unnormalized Laplacian or not',
        action='store_true'
        )
parser.add_argument(
        '-lar', '--largest',
        help='whether to use largest (Rohe) or smallest (von Luxburg) k\
        eigenvectors of the Laplacian.', action='store_true'
        )
parser.add_argument(
        '-l', '--load',
        help='Loads saved graph', type=str)
args = parser.parse_args()


def main():
    # Generate SBM graph
    if args.load is None:
        Gamma = np.array([
            [0.4, 0.4, 0.01],
            [0.4, 0.4, 0.01],
            [0.01, 0.01, 0.4]
            ])
        Pi = np.array(3*[1/3])
        model = SBM(args.n, Gamma, Pi)
        Z, Z_v, A = model.sample()
    else:
        Gamma, Pi, Z, Z_v, A = load_graph(args.load)

    if args.visual:
        draw_graph(A, Z_v)

    # Run the spectral clustering algorithm
    spectral_clustering = SpectralClustering(args.k)
    labels, eigvecs, eigvals, centroids = spectral_clustering.cluster(
            A, args.unnormalized, args.largest)
    accuracy_value, accuracy_permutation = accuracy(labels, Z_v)
    visualize_eigenvectors(eigvecs, centroids, labels, accuracy_permutation)
    visualize_eigenvalues(eigvals)
    print('Accuracy: ', accuracy_value)


if __name__=="__main__":
    main()

