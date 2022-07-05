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
args = parser.parse_args()


def main():
    # Generate SBM graph
    Gamma = np.array([
            [1, 0.0],
            [0.0, 1]
            ])
    Pi = np.array([0.5, 0.5])
    model = SBM(args.n, Gamma, Pi)
    Z, Z_v, A = model.sample()

    # Run the spectral clustering algorithm
    spectral_clustering = SpectralClustering(args.k)
    labels, centroids = spectral_clustering.cluster(A, Z_v)
    accuracy_value, accuracy_permutation = accuracy(labels, Z_v)
    print('Accuracy: ', accuracy_value)

    # Compare with sklearn
    skspectralclustering = skSpectralClustering(
            n_clusters=2,
            affinity='precomputed').fit(A)
    sklabels = skspectralclustering.labels_
    skaccuracy_value, _ = accuracy(sklabels, Z_v)
    print('SK accuracy: ', skaccuracy_value)


if __name__=="__main__":
    main()

