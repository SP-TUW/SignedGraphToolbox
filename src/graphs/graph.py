from scipy.sparse import csr_matrix
from scipy.sparse import diags

class Graph:

    def __init__(self, w, w_neg=None):
        '''

        :param w: weight matrix of the graph. Can be any structure that scipy.sparse can handle. If w_neg is specified then only the positive part will be used to define w_pos, otherwise w will be split into w_pos and w_neg.
        :param w_neg: weight matrix of the negative part of the graph. Numeric values need to be positive.
        '''
        self.w = csr_matrix(w)
        if w_neg is not None:
            self.w_pos = csr_matrix(w).maximum(0)
            self.w_neg = csr_matrix(w_neg).maximum(0)
        else:
            self.w_pos = csr_matrix(w).maximum(0)
            self.w_neg = csr_matrix(-w).maximum(0)

    def get_signed_laplacian(self):
        degree = self.w_pos.sum(1) + self.w_neg.sum(1)
        lap = diags(degree) - self.w_pos + self.w_neg
        return lap

    def
