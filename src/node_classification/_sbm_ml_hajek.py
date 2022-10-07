'''
Implementation of the spectral method of
    Hajek, B., Wu, Y., & Xu, J. (2016).
    Achieving exact cluster recovery threshold via semidefinite programming: Extensions.
    IEEE Transactions on Information Theory, 62(10), 5918-5937.
for the ML estimation of cluster association for the stochastic block model (SBM).
'''

import numpy as np
import scipy as sc
import cvxpy as cp
from sklearn.cluster import KMeans

from ._node_learner import NodeLearner


class SbmMlHajek(NodeLearner):

    def __init__(self, num_classes=2, class_distribution=None, verbosity=0, save_intermediate=False, kmeans_args={}):
        self.kmeans_args = kmeans_args

        if class_distribution is None:
            self.class_distribution = np.ones(num_classes) / num_classes
        else:
            self.class_distribution = np.array(class_distribution) / sum(class_distribution)

        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        num_nodes = graph.num_nodes


        X = cp.Variable(shape=(num_nodes, num_nodes), name='X', PSD=True)

        A = graph.weights.maximum(0).A
        objective_expression = cp.scalar_product(A, X)
        objective = cp.Maximize(objective_expression)

        constraints = [cp.diag(X) == 1,
                       -X <= 0,
                       cp.sum(X) == (num_nodes * np.linalg.norm(self.class_distribution)) ** 2]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        Y = problem.variables()[0].value
        (d, x) = np.linalg.eigh(Y)

        kmeans = KMeans(n_clusters=self.num_classes)
        l_est = kmeans.fit_predict(x[:, -self.num_classes:], **self.kmeans_args)
        self.embedding = x[:, -self.num_classes:]
        return l_est
