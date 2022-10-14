from abc import ABC

from scipy.sparse import csr_matrix
from scipy.sparse import diags
import numpy as np


class Graph(ABC):

    def __init__(self, num_classes, class_labels, weights, weights_neg=None):
        '''

        :param num_classes: ground truth number of classes
        :param class_labels: ground truth class labels
        :param weights: weight matrix of the graph. Can be any structure that scipy.sparse can handle. If weights_neg is specified then only the positive part will be used to define w_pos, otherwise w will be split into w_pos and weights_neg.
        :param weights_neg: weight matrix of the negative part of the graph. Numeric values need to be positive.
        '''
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

    def get_signed_laplacian(self):
        lap = diags(self.degree) - self.w_pos + self.w_neg
        return lap

    def get_signed_sym_laplacian(self):

        lap = diags(self.degree) - self.w_pos + self.w_neg
        inv_sqrt_deg = np.array([1/d if d>0 else 0 for d in np.sqrt(self.degree)])
        lap = diags(inv_sqrt_deg).dot(lap).dot(diags(inv_sqrt_deg))

        return lap

    def get_signed_am_laplacian(self):

        lap_pos = diags(self.d_pos) - self.w_pos
        inv_sqrt_pos_deg = np.array([1/d if d>0 else 0 for d in np.sqrt(self.d_pos)])
        lap_pos = diags(inv_sqrt_pos_deg).dot(lap_pos).dot(diags(inv_sqrt_pos_deg))

        lap_neg = diags(self.d_neg) + self.w_neg
        inv_sqrt_neg_deg = np.array([1/d if d>0 else 0 for d in np.sqrt(self.d_neg)])
        lap_neg = diags(inv_sqrt_neg_deg).dot(lap_neg).dot(diags(inv_sqrt_neg_deg))

        lap = 1/2 * (lap_pos + lap_neg)

        return lap

