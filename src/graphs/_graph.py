from abc import ABC

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags


class Graph(ABC):

    def __init__(self, num_classes, class_labels, weights, weights_neg=None, name=None):
        '''

        :param num_classes: ground truth number of classes
        :param class_labels: ground truth class labels
        :param weights: weight matrix of the graph. Can be any structure that scipy.sparse can handle. If weights_neg is specified then only the positive part will be used to define w_pos, otherwise w will be split into w_pos and weights_neg.
        :param weights_neg: weight matrix of the negative part of the graph. Numeric values need to be positive.
        '''
        self.name = name
        self.num_nodes = weights.shape[0]
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.weights = csr_matrix(weights)
        self.weights.eliminate_zeros()
        if weights_neg is not None:
            self.w_pos = csr_matrix(weights).maximum(0)
            self.w_neg = csr_matrix(weights_neg).maximum(0)
        else:
            self.w_pos = csr_matrix(weights).maximum(0)
            self.w_neg = csr_matrix(-weights).maximum(0)
        self.d_pos = np.squeeze(np.asarray(self.w_pos.sum(1)))
        self.d_neg = np.squeeze(np.asarray(self.w_neg.sum(1)))
        self.degree = self.d_pos + self.d_neg
        self._gradient_matrix = {}

    def get_pos_laplacian(self):
        '''

        :return: laplacian of positive edges
        '''
        lap_pos = diags(self.d_pos) - self.w_pos
        return lap_pos

    def get_pos_sym_laplacian(self):
        '''

        :return: symmetric laplacian of positive edges
        '''
        lap_pos = self.get_pos_laplacian()
        inv_sqrt_pos_deg = np.array([1/d if d>0 else 0 for d in np.sqrt(self.d_pos)])
        lap_pos = diags(inv_sqrt_pos_deg).dot(lap_pos).dot(diags(inv_sqrt_pos_deg))
        return lap_pos

    def get_neg_laplacian(self):
        '''

        :return: laplacian of negative edges
        '''
        lap_neg = diags(self.d_neg) + self.w_neg
        return lap_neg

    def get_neg_sym_laplacian(self):
        '''

        :return: symmetric laplacian of negative edges
        '''
        lap_neg = self.get_neg_laplacian()
        inv_sqrt_neg_deg = np.array([1/d if d>0 else 0 for d in np.sqrt(self.d_neg)])
        lap_neg = diags(inv_sqrt_neg_deg).dot(lap_neg).dot(diags(inv_sqrt_neg_deg))
        return lap_neg

    def get_signed_laplacian(self):
        '''

        :return: signed laplacian
        '''
        lap = diags(self.degree) - self.w_pos + self.w_neg
        return lap

    def get_signed_sym_laplacian(self):
        '''

        :return: symmetric normalized signed laplacian
        '''
        lap = diags(self.degree) - self.w_pos + self.w_neg
        inv_sqrt_deg = np.array([1/d if d>0 else 0 for d in np.sqrt(self.degree)])
        lap = diags(inv_sqrt_deg).dot(lap).dot(diags(inv_sqrt_deg))

        return lap

    def get_signed_am_laplacian(self):
        '''

        :return: arithmetic mean laplacian
        '''

        lap_pos = self.get_pos_sym_laplacian()
        lap_neg = self.get_neg_sym_laplacian()

        lap = 1/2 * (lap_pos + lap_neg)

        return lap

    def get_sponge_matrices(self, tau_sim=1, tau_dis=1):
        matrix_numerator = diags(self.d_pos) - self.w_pos + tau_dis*diags(self.d_neg)
        matrix_denominator = diags(self.d_neg) - self.w_neg + tau_sim*diags(self.d_pos)
        return matrix_numerator, matrix_denominator

    def get_gradient_matrix(self, p, return_div=False):
        if p not in self._gradient_matrix:
            num_edges = self.weights.data.size
            num_nodes = self.weights.shape[0]
            weights_coo = self.weights.tocoo()
            i = np.arange(num_edges)
            i = np.tile(i, 2)
            j = np.r_[weights_coo.row, weights_coo.col]
            v = np.power(np.abs(weights_coo.data), 1 / p)
            v = np.concatenate((v, -np.sign(weights_coo.data) * v))
            self._gradient_matrix[p] = csr_matrix((v, (i, j)), shape=(num_edges, num_nodes))
        if not return_div:
            return self._gradient_matrix[p]
        else:
            divergence_matrix = -self._gradient_matrix[p].T
            return self._gradient_matrix[p], divergence_matrix

