from abc import ABC

from scipy.sparse import csr_matrix
from scipy.sparse import diags


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

    def get_signed_laplacian(self):
        degree = self.w_pos.sum(1) + self.w_neg.sum(1)
        lap = diags(degree) - self.w_pos + self.w_neg
        return lap

