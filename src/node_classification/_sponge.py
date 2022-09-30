# This is a wrapper for the code from https://github.com/alan-turing-institute/SigNet
# it is necessary to install SigNet via
# pip install git+https://github.com/alan-turing-institute/SigNet.git

from signet.cluster import Cluster
import numpy as np
from scipy import sparse as ss

from ._node_learner import NodeLearner



class Sponge(NodeLearner):
    def __init__(self, num_classes=2, objective=None, verbosity=0, save_intermediate=False, **kwargs):
        self.objective = objective
        self.sponge_kwargs = kwargs
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        a_p = graph.weights.maximum(0).tocsc()
        a_n = -graph.weights.minimum(0).tocsc()

        c = Cluster((a_p, a_n))

        if self.objective is None:
            l_pred = c.SPONGE(self.num_classes)
        elif self.objective == 'SYM':
            l_pred = c.SPONGE_sym(self.num_classes)
        elif self.objective == 'BNC':
            l_pred = c.spectral_cluster_bnc(self.num_classes, **self.sponge_kwargs)

        return l_pred
