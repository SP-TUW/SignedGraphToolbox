# Implementation of (4) from
# Zhu, X., Ghahramani, Z., & Lafferty, J. D. (2003).
# Semi-supervised learning using gaussian fields and harmonic functions.
# In Proceedings of the 20th International conference on Machine learning (ICML-03) (pp. 912-919).

# Implemented by Thomas Dittrich 2020

import numpy as np
from scipy.sparse.linalg import lsqr, spsolve
from scipy.sparse import diags
from ._node_learner import NodeLearner


class HarmonicFunctions(NodeLearner):
    def __init__(self, num_classes=2, verbosity=1, save_intermediate=None, class_prior=1/2):
        self.class_prior = class_prior
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        if self.num_classes > 2:
            raise ValueError('more than 2 clusters currenly not supported')
        fl = np.array(labels['k'])
        labelled_nodes = labels['i']
        unlabelled_nodes = [i for i in range(graph.num_nodes) if i not in labelled_nodes]

        W = graph.weights.maximum(0)
        D = np.array(W.sum(1)).flatten()
        L = diags(D)-W

        laplacian_uu = L[unlabelled_nodes, :]
        laplacian_uu = laplacian_uu[:, unlabelled_nodes]
        w_ul = W[unlabelled_nodes, :]
        w_ul = w_ul[:, labelled_nodes]
        fu = lsqr(laplacian_uu, w_ul.dot(fl), atol=1e-8, btol=1e-8)[0]
        # fu = spsolve(laplacian_uu, w_ul.dot(fl))
        f = np.zeros(graph.num_nodes)
        f[labelled_nodes] = fl
        f[unlabelled_nodes] = fu

        class_mass = [np.sum(1-fu), np.sum(fu)]

        # label_estimates = np.maximum(np.minimum(np.round(f), 1), 0).astype(int)
        label_estimates = self.class_prior*f/class_mass[1] > (1-self.class_prior)*(1-f)/class_mass[0]
        label_estimates = label_estimates.astype(int)
        label_estimates[labels['i']] = labels['k']
        self.embedding = f
        self.normalized_embedding = f
        return label_estimates
