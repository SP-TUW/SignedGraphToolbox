import warnings

import numpy as np

from src.tools.projections import simplex_projection
from ._node_learner import NodeLearner


def objective(x_in, cluster_distribution, edge_probability, weights, use_quadratic=False):
    num_nodes = x_in.shape[0]
    if use_quadratic:
        x_out = -np.sum(x_in.dot(np.log(cluster_distribution)))
        for i in range(num_nodes):
            j = [j_ for j_ in range(num_nodes) if j_ != i]
            pij = edge_probability[:, weights[i, j], :]
            px = np.sum(np.log(pij) * x_in[j, :], 2)
            xpx = x_in[i, :].dot(px)
            x_out = x_out - np.sum(xpx) / 2
    else:
        x_out = -np.sum(np.log(x_in.dot(cluster_distribution)))
        for i in range(num_nodes):
            j = [j_ for j_ in range(num_nodes) if j_ != i]
            pij = edge_probability[:, weights[i, j], :]
            px = np.sum(pij * x_in[j, :], 2)
            xpx = x_in[i, :].dot(px)
            x_out = x_out - np.sum(np.log(xpx)) / 2
    return x_out


def projection(x_in):
    x_out = simplex_projection(x_in)
    # x_out = x_in/np.linalg.norm(x_in, axis=1, keepdims=True)
    return x_out


def gradient(x_in, cluster_distribution, edge_probability, weights, use_quadratic=False):
    num_nodes = x_in.shape[0]
    num_clusters = cluster_distribution.size

    if use_quadratic:
        x_out = np.ones((num_nodes,num_clusters))*np.log(cluster_distribution)
        for m in range(num_nodes):
            j = [i for i in range(num_nodes) if i != m]
            pmj = edge_probability[:, weights[m, j], :]
            px = np.sum(np.log(pmj) * x_in[j, :], 2)
            x_out[m, :] = x_out[m, :] - np.sum(px, 1)

    else:
        x_out = np.outer(-1 / (x_in.dot(cluster_distribution)), cluster_distribution)
        for m in range(num_nodes):
            j = [i for i in range(num_nodes) if i != m]
            pmj = edge_probability[:, weights[m, j], :]
            px = np.sum(pmj * x_in[j, :], 2)
            xpx = np.maximum(x_in[m, :].dot(px), 1e-3 / num_nodes)
            x_out[m, :] = x_out[m, :] - np.sum(px / xpx, 1)

    return x_out


def calc_edge_probabilities(pi, pe, li, le, num_clusters=2):
    p0 = 1 - pi
    pp1 = pi * (1 - li)
    pm1 = pi * li
    q0 = 1 - pe
    qp1 = pe * (1 - le)
    qm1 = pe * le

    ep0 = (p0 - q0) * np.eye(num_clusters) + q0 * np.ones((num_clusters, num_clusters))
    epp1 = (pp1 - qp1) * np.eye(num_clusters) + qp1 * np.ones((num_clusters, num_clusters))
    epm1 = (pm1 - qm1) * np.eye(num_clusters) + qm1 * np.ones((num_clusters, num_clusters))
    edge_probability = np.zeros((num_clusters, 3, num_clusters))
    edge_probability[:, 0, :] = ep0
    edge_probability[:, 1, :] = epp1
    edge_probability[:, 2, :] = epm1
    return edge_probability


class LSBMMAP(NodeLearner):
    def __init__(self, num_classes=2, verbosity=0, save_intermediate=False, eps=1e-5, t_max=1e5):
        self.num_classes = num_classes
        self.objective = objective
        self.verbosity = verbosity
        self.eps = eps
        self.t_max = t_max
        self.save_intermediate = save_intermediate
        self.intermediate_results = None
        self.embedding = None
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):

def cluster(graph, pi, pe, li, le, num_clusters=2, cluster_distribution=None, x0=None,
            eps_stop=1e-4, t_max=10000, use_quadratic=False, verbosity=1, **kwargs):
    num_nodes = graph.N

    if cluster_distribution is None:
        cluster_distribution = np.ones(num_clusters) / num_clusters
    else:
        cluster_distribution = np.array(cluster_distribution) / sum(cluster_distribution)
    edge_probability = calc_edge_probabilities(pi=pi, pe=pe, li=li, le=le, num_clusters=num_clusters)

    weights = graph.W.A.astype('int8')

    if x0 is None:
        x0 = np.ones((num_nodes, num_clusters))
        x0 = x0 + 1e-1 * np.random.randn(num_nodes, num_clusters)
    else:
        x0 = x0 * np.ones((num_nodes, num_clusters))

    x_new = projection(x0)
    t = 0
    tau0 = 0.1
    f_x_new = objective(x_new, cluster_distribution, edge_probability, weights, use_quadratic=use_quadratic)
    converged = False
    while not converged:
        x = x_new.copy()
        g_x = gradient(x, cluster_distribution, edge_probability, weights, use_quadratic=use_quadratic)
        direction = projection(x - g_x) - x
        if np.any(np.isnan(direction)):
            raise ValueError('Algorithm ended up in non-differentiable situation. Check input and initial value!')
        slope = np.sum(g_x*direction)
        f_x = f_x_new
        tau = tau0
        backtracking_converged = False
        while not backtracking_converged:
            x_new = x + tau * direction
            f_x_new = objective(x_new, cluster_distribution, edge_probability, weights, use_quadratic=use_quadratic)
            if f_x - f_x_new >= -tau * slope * 1/2:
                backtracking_converged = True
            else:
                tau = tau/2
        if verbosity > 0 and t % 10 == 0:
            print('{t:4d}, {n:e}'.format(t=t, n=np.linalg.norm(direction)))
        t = t + 1
        converged = np.linalg.norm(direction) < np.sqrt(num_nodes * num_clusters) * eps_stop
        if not converged and t > t_max:
            warnings.warn("Algorithm did not converge within {t} steps".format(t=t_max))
            break

    l_est = np.argmax(x, 1)
    return x_new, l_est