import warnings

import numpy as np

from src.tools.projections import simplex_projection, label_projection
from ._node_learner import NodeLearner


def _objective(x_in, class_distribution, edge_probability, weights, use_quadratic=False):
    num_nodes = x_in.shape[0]
    if use_quadratic:
        x_out = -np.sum(x_in.dot(np.log(class_distribution)))
        for i in range(num_nodes):
            j = [j_ for j_ in range(num_nodes) if j_ != i]
            pij = edge_probability[:, weights[i, j], :]
            px = np.sum(np.log(pij) * x_in[j, :], 2)
            xpx = x_in[i, :].dot(px)
            x_out = x_out - np.sum(xpx) / 2
    else:
        x_out = -np.sum(np.log(x_in.dot(class_distribution)))
        for i in range(num_nodes):
            j = [j_ for j_ in range(num_nodes) if j_ != i]
            pij = edge_probability[:, weights[i, j], :]
            px = np.sum(pij * x_in[j, :], 2)
            xpx = x_in[i, :].dot(px)
            x_out = x_out - np.sum(np.log(xpx)) / 2
    return x_out


def _projection(x_in, labels):
    x_simplex = simplex_projection(x_in)
    x_out = label_projection(x_simplex, labels)
    return x_out


def _gradient(x_in, class_distribution, edge_probability, weights, use_quadratic=False):
    num_nodes = x_in.shape[0]
    num_classes = class_distribution.size

    if use_quadratic:
        x_out = np.ones((num_nodes,num_classes))*np.log(class_distribution)
        for m in range(num_nodes):
            j = [i for i in range(num_nodes) if i != m]
            pmj = edge_probability[:, weights[m, j], :]
            px = np.sum(np.log(pmj) * x_in[j, :], 2)
            x_out[m, :] = x_out[m, :] - np.sum(px, 1)

    else:
        x_out = np.outer(-1 / (x_in.dot(class_distribution)), class_distribution)
        for m in range(num_nodes):
            j = [i for i in range(num_nodes) if i != m]
            pmj = edge_probability[:, weights[m, j], :]
            px = np.sum(pmj * x_in[j, :], 2)
            xpx = np.maximum(x_in[m, :].dot(px), 1e-3 / num_nodes)
            x_out[m, :] = x_out[m, :] - np.sum(px / xpx, 1)

    return x_out


def _calc_edge_probabilities(pi, pe, li, le, num_classes=2):
    p0 = 1 - pi
    pp1 = pi * (1 - li)
    pm1 = pi * li
    q0 = 1 - pe
    qp1 = pe * (1 - le)
    qm1 = pe * le

    ep0 = (p0 - q0) * np.eye(num_classes) + q0 * np.ones((num_classes, num_classes))
    epp1 = (pp1 - qp1) * np.eye(num_classes) + qp1 * np.ones((num_classes, num_classes))
    epm1 = (pm1 - qm1) * np.eye(num_classes) + qm1 * np.ones((num_classes, num_classes))
    edge_probability = np.zeros((num_classes, 3, num_classes))
    edge_probability[:, 0, :] = ep0
    edge_probability[:, 1, :] = epp1
    edge_probability[:, 2, :] = epm1
    return edge_probability


class LsbmMap(NodeLearner):
    def __init__(self, pi, pe, li, le, num_classes=2, class_distribution=None, verbosity=0, save_intermediate=False, eps=1e-5, t_max=1e5):
        self.eps = eps
        self.t_max = t_max
        self.use_quadratic = False

        if class_distribution is None:
            self.class_distribution = np.ones(num_classes) / num_classes
        else:
            self.class_distribution = np.array(class_distribution) / sum(class_distribution)
        self.edge_probability = _calc_edge_probabilities(pi=pi, pe=pe, li=li, le=le, num_classes=num_classes)
        
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        num_nodes = graph.num_nodes
        weights = graph.weights.A.astype('int8')

        if guess is None:
            x0 = np.ones((num_nodes, self.num_classes))
            x0 = x0 + 1e-1 * np.random.randn(num_nodes, self.num_classes)
            x0 = _projection(x0, labels)
        else:
            x0 = 0.45 * np.ones((num_nodes, self.num_classes))
            x0[range(num_nodes),guess] = 0.55
            x0 = _projection(x0, labels)

        x_new = _projection(x0, labels)
        t = 0
        tau0 = 0.1
        f_x_new = _objective(x_new, self.class_distribution, self.edge_probability, weights, use_quadratic=self.use_quadratic)
        converged = False
        while not converged:
            x = x_new.copy()
            g_x = _gradient(x, self.class_distribution, self.edge_probability, weights, use_quadratic=self.use_quadratic)
            direction = _projection(x - g_x, labels) - x
            if np.any(np.isnan(direction)):
                raise ValueError('Algorithm ended up in non-differentiable situation. Check input and initial value!')
            slope = np.sum(g_x*direction)
            f_x = f_x_new
            tau = tau0
            backtracking_converged = False
            while not backtracking_converged:
                x_new = x + tau * direction
                f_x_new = _objective(x_new, self.class_distribution, self.edge_probability, weights, use_quadratic=self.use_quadratic)
                if f_x - f_x_new >= -tau * slope * 1/2:
                    backtracking_converged = True
                else:
                    tau = tau/2
            if self.verbosity > 0 and t % 10 == 0:
                print('{t:4d}, {n:e}, {o:e}'.format(t=t, n=np.linalg.norm(direction), o=f_x_new))
            t = t + 1
            converged = np.linalg.norm(direction) < np.sqrt(num_nodes * self.num_classes) * self.eps and t>1
            if not converged and t > self.t_max:
                warnings.warn("Algorithm did not converge within {t} steps".format(t=self.t_max))
                break

        self.embedding = x_new
        self.normalized_embedding = x_new
        l_est = np.argmax(x, 1)
        return l_est