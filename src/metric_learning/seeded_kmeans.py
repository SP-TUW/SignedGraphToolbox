import numpy as np
from scipy.spatial.distance import cdist
import warnings

def cluster_with_centroids(X, num_clusters, labels, t_max, eps):
    X = X.copy()
    N = X.shape[0]
    d = X.shape[1]
    c = np.zeros((num_clusters, d))
    labels_i = np.array(labels['i'])
    labels_k = np.array(labels['k'])
    for k in range(num_clusters):
        i_labels_in_k = labels_i[labels_k == k]
        c[k, :] = np.mean(X[i_labels_in_k, :], axis=0)

    c_new = np.zeros((num_clusters, d))
    converged = False
    t = 0
    while not converged:
        t += 1
        dist = cdist(X, c, 'euclidean')
        l = np.argmin(dist, 1)
        l[labels['i']] = labels['k']
        for j in range(num_clusters):
            c_new[j] = np.mean(X[l == j, :], 0)
        if np.linalg.norm(c - c_new) < eps * np.sqrt(X.size):
            converged = True
        if t >= t_max and not converged:
            warnings.warn('kmeans did not converge')
            break
        c = c_new
    return l, c