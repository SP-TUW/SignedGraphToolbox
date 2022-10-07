'''
Implementation of the spectral method of
    Lelarge, M., Massoulié, L., & Xu, J. (2015).
    Reconstruction in the labelled stochastic block model.
    IEEE Transactions on Network Science and Engineering, 2(4), 152-163.
for the ML estimation of cluster association for the labelled stochastic block model (SBM).

This implementation only consider labels in {-1,1}
'''

import numpy as np
import scipy as sc

from ._node_learner import NodeLearner


def get_weight_function(edge_labels, a, b, li, le):
    '''
    This function implements the optimal weight function from Theorem 1 for edge labels in {-1,1} for N nodes
    :param edge_labels: [NxN] matrix containing the edge labels. Zero entries mean that no edge is present
    :param a: Mean internal degree of the SBM
    :param b: Mean external degree of the SBM
    :param li: Probability that an internal edge has label -1
    :param le: Probability that an external edge has label -1
    :return: Weight matrix
    '''
    mu = np.array([b, (1 - li), li])
    nu = np.array([a, (1 - le), le])
    weight_function = (a * mu - b * nu) / (a * mu + b * nu)
    mu[0] = 0
    nu[0] = 0
    return weight_function, mu, nu


class LsbmMlLelarge(NodeLearner):
    def __init__(self, pi, pe, li, le, num_classes=2, verbosity=0, save_intermediate=False):
        assert num_classes == 2, 'LsbmMlLelarge only supports 2 clusters'
        self.pi = pi
        self.pe = pe
        self.li = li
        self.le = le
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):

        if labels is not None:
            print('LsbmMlLelarge does not support labels')

        num_nodes = graph.num_nodes
        edge_labels = graph.weights.A.astype('int')
        assert np.all(edge_labels == edge_labels.T), 'edge labels need to be symmetric'

        a = num_nodes * self.pi
        b = num_nodes * self.pe
        weight_function, mu, nu = get_weight_function(edge_labels, a, b, self.li, self.le)
        alpha = 1 / 2 * (weight_function[+1] * (a * mu[+1] + b * nu[+1]) +
                         weight_function[-1] * (a * mu[-1] + b * nu[-1]))
        weights = weight_function[edge_labels]
        d_matrix = weights - alpha / num_nodes

        v, x = sc.sparse.linalg.eigsh(d_matrix, k=1, which='LM')

        self.embedding = x
        l_est = np.squeeze(x >= 0).astype('int')
        return l_est
