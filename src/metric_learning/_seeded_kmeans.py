import warnings

import numpy as np
from scipy.spatial.distance import cdist

from ._metric_learner import MetricLearner


class SeededKMeans(MetricLearner):

    def __init__(self, num_classes, verbose=0, save_intermediate=False, t_max=200, eps=1e-5):
        self.t_max = t_max
        self.eps = eps
        self.class_centers = None
        super().__init__(num_classes=num_classes, verbose=verbose, save_intermediate=save_intermediate)

    def estimate_labels(self, data, labels):
        data = data.copy()
        N = data.shape[0]
        d = data.shape[1]
        class_centers = np.zeros((self.num_classes, d))
        labels_i = np.array(labels['i'])
        labels_k = np.array(labels['k'])
        for k in range(self.num_classes):
            i_labels_in_k = labels_i[labels_k == k]
            class_centers[k, :] = np.mean(data[i_labels_in_k, :], axis=0)

        c_new = np.zeros((self.num_classes, d))
        converged = False
        t = 0
        while not converged:
            t += 1
            dist = cdist(data, class_centers, 'euclidean')
            l_est = np.argmin(dist, 1)
            l_est[labels['i']] = labels['k']
            for j in range(self.num_classes):
                c_new[j] = np.mean(data[l_est == j, :], 0)
            if np.linalg.norm(class_centers - c_new) < self.eps * np.sqrt(data.size):
                converged = True
            if t >= self.t_max and not converged:
                warnings.warn('kmeans did not converge')
                break
            class_centers = c_new
        self.l_est = l_est
        self.class_centers = class_centers
        return l_est