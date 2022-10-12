# TV minimization from:
# Bresson, Xavier, et al. "Multiclass total variation clustering."
# Advances in Neural Information Processing Systems. 2013.

# Implemented by Thomas Dittrich 2020

import numpy as np
from src.tools.projections import simplex_projection
import scipy.sparse as sps
import warnings

from src.node_classification._node_learner import NodeLearner


def norm_1la(t, la):
    #  calculates the 1,lambda-norm along dim 0
    i_p = np.where(t > 0)
    i_n = np.where(t < 0)
    t[i_p] = la * t[i_p]
    t[i_n] = -t[i_n]
    return np.sum(t, 0, keepdims=True)


def med_la(f_in, la):
    #  calculates the lambda-median along dim 0
    num_nodes = f_in.shape[0]
    k = int(np.floor(num_nodes / (la + 1)))
    f_sort = np.sort(f_in, axis=0)
    # paper says the (k+1)-st largest value but in their Matlab code it was the (k+1)-st element from the ascending
    # ordering
    return f_sort[(k), ...][np.newaxis, ...]


def func_b(f_in, la):
    num_nodes = f_in.shape[0]
    med_val = med_la(f_in, la)
    b = norm_1la(f_in - np.repeat(med_val, num_nodes, axis=0), la)
    return b


def func_db(f_in, la):
    med = med_la(f_in, la)
    v = np.zeros(f_in.shape)
    for k in range(f_in.shape[1]):
        i_m = f_in[:, k] < med[0, k]
        i_0 = f_in[:, k] == med[0, k]
        i_p = f_in[:, k] > med[0, k]
        n_m = np.sum(i_m, 0)
        n_0 = np.sum(i_0, 0)
        n_p = np.sum(i_p, 0)
        v[i_m, k] = -la
        v[i_0, k] = (la * n_m - n_p) / n_0
        v[i_p, k] = 1
    return v


def func_t(f_in, gradient_matrix):
    grad = np.sum(np.abs(gradient_matrix * f_in), 0)
    return grad


def proj_c(f_in, labels):
    f_out = simplex_projection(f_in)
    f_out[labels['i'], :] = 0
    f_out[labels['i'], labels['k']] = 1
    return f_out


def _run_minimization(graph, num_classes, labels, t_outer_max, t_inner_max, eps_outer, eps_inner, save_intermediate):
    num_nodes = graph.num_nodes
    weights = graph.weights.tocoo()
    weights.data[weights.data < 0] = 0
    weights.eliminate_zeros()

    num_edges = weights.nnz

    if labels is None:
        labels = {'i': [], 'k': []}

    la = 1 / (num_classes - 1)

    #  gradient matrix: K in pseudocode
    gradient_matrix = np.zeros((num_edges, num_nodes))
    gradient_matrix = sps.csr_matrix(
        (np.r_[weights.data, -weights.data], (np.r_[np.arange(num_edges), np.arange(num_edges)], np.r_[weights.row, weights.col])),
        shape=(num_edges, num_nodes), dtype=float)
    gradient_matrix[range(num_edges), weights.row] = -weights.data
    gradient_matrix[range(num_edges), weights.col] = weights.data

    f = proj_c(np.ones((num_nodes, num_classes)) / num_classes, labels)
    for (i, k) in zip(labels['i'], labels['k']):
        f[i, :] = 0
        f[i, k] = 1
    p = np.zeros((num_edges, num_classes))

    ATA = gradient_matrix.T.dot(gradient_matrix)
    l = np.sqrt(sps.linalg.eigsh(ATA, k=1, which='LM', return_eigenvectors=False))[0]
    tau = 1 / l
    eps = 1e-6
    b_r = func_b(f, la)
    t_r = func_t(f, gradient_matrix)
    e_r = t_r / b_r
    t_outer = 0
    outer_loop_converged = False
    while not outer_loop_converged:
        b_r_km1 = b_r.copy()
        e_r_km1 = e_r.copy()
        fkm1 = f.copy()
        delta = np.max(b_r)
        delta_0 = np.min(b_r)
        sigma = delta_0 ** 2 / (tau * delta ** 2 * l ** 2)
        f_bar = f.copy()
        d_e = np.diag((e_r / b_r)[0, :])
        d_b = np.diag((delta / b_r)[0, :])
        v = delta * func_db(f, la).dot(d_e)
        g = f + v
        inner_loop_converged = False
        t_inner = 0
        while not inner_loop_converged:
            f_old = f.copy()
            p_tilde = p + sigma * gradient_matrix.dot(f_bar).dot(d_b)
            p = p_tilde / np.maximum(np.abs(p_tilde), 1)
            f_tilde = f - tau * (gradient_matrix.T * p).dot(d_b)
            f = (f_tilde + tau * g) / (1 + tau)
            f = proj_c(f, labels)
            theta = 1 / np.sqrt(1 + 2 * tau)
            tau = theta * tau
            sigma = sigma / theta
            f_bar = (1 + theta) * f - theta * f_old
            b_r = func_b(f, la)
            t_r = func_t(f, gradient_matrix)
            e_r = t_r / b_r
            c1 = np.sum(b_r / b_r_km1 * (e_r_km1 - e_r))
            c2 = (1 - eps_inner) * np.linalg.norm(fkm1 - f, ord='fro') ** 2 / delta
            t_inner = t_inner + 1
            inner_loop_converged = (c1 >= c2) or t_inner > t_inner_max  # loop continues as long as this fails
        t_outer = t_outer + 1
        outer_loop_converged = np.linalg.norm(fkm1 - f, ord='fro') < np.sqrt(num_nodes * num_classes) * eps_outer
        if t_outer > t_outer_max:
            warnings.warn("Algorithm did not converge within {t} steps".format(t=t_outer_max))
            break
    l_est = np.argmax(f, axis=1)
    intermediate_results = {}
    return l_est, f, intermediate_results


class TvBresson(NodeLearner):
    def __init__(self, num_classes=2, verbosity=0, save_intermediate=None, t_outer_max=10000, t_inner_max=100,
                 eps_outer=1e-5, eps_inner=1e-6):
        self.t_outer_max = t_outer_max
        self.t_inner_max = t_inner_max
        self.eps_outer = eps_outer
        self.eps_inner = eps_inner
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        l_est, x, intermediate_results = _run_minimization(graph=graph, num_classes=self.num_classes, labels=labels,
                                                           t_outer_max=self.t_outer_max, t_inner_max=self.t_inner_max,
                                                           eps_outer=self.eps_outer, eps_inner=self.eps_inner,
                                                           save_intermediate=self.save_intermediate)
        self.embedding = x
        return l_est
