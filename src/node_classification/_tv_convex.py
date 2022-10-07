from src.node_classification._node_learner import NodeLearner

import numpy as np
from src.tools.projections import simplex_projection, label_projection, min_norm_simplex_projection
from numpy import sqrt
from numpy.linalg import norm
from numpy.random import randn
from scipy import sparse as sps
import warnings


def qProjection(X, labels):
    X_ = simplex_projection(X + 1, 2) - 1
    X_ = label_projection(X_, labels)
    return X_


def gradient(W, X, p=1):
    assert sps.isspmatrix_csr(W)
    num_nodes = W.shape[0]
    w_row_sliced = (np.sign(W.data)*np.abs(W.data) ** (1 / p))[:, np.newaxis]
    i = np.repeat(np.arange(num_nodes), np.diff(W.indptr))
    j = W.indices
    WaXi = X[i, :] * np.abs(w_row_sliced)
    WXj = X[j, :] * w_row_sliced
    return WaXi - WXj


def divergence(W, Z, column_slicing_permutation, column_slicing_indptr, p=1):
    num_nodes = W.shape[0]
    num_clusters = Z.shape[1]
    w_row_sliced = (np.sign(W.data)*np.abs(W.data) ** (1 / p))[:, np.newaxis]
    X = np.zeros((num_nodes, num_clusters))
    # calculate W*Z and arange such that elements represent stacked columns of W
    WZ_full = (w_row_sliced * Z)[column_slicing_permutation, :]
    # sum over all elements that belong to the same column of W
    i = [i for i in column_slicing_indptr if i < column_slicing_indptr[-1]]
    WZ = np.zeros((num_nodes, num_clusters))
    WZ[range(len(i)), :] = np.add.reduceat(WZ_full, i, axis=0)
    # calculate |W|*Z (elements represent stacked rows of W)
    WaZ_full = np.abs(w_row_sliced) * Z
    # sum over all elements that belong to the same row of W
    i = [i for i in W.indptr[:-1] if i < W.indptr[-1]]
    WaZ = np.zeros((num_nodes, num_clusters))
    WaZ[range(len(i)), :] = np.add.reduceat(WaZ_full, i, axis=0)
    return WZ - WaZ


def TV(W, X, p=1):
    return np.sum(np.maximum(gradient(W, X, p=p), 0) ** p)


def get_slicing_permutations(W):
    num_nodes = W.shape[0]
    column_slicing_permutation = np.argsort(W.indices)
    sorted_indices = np.sort(W.indices)
    unique_sorted_indices = np.unique(sorted_indices)
    column_slicing_indptr_temp = np.r_[0, np.where(np.diff(sorted_indices))[0] + 1]
    column_slicing_indptr = W.indices.size * np.ones(num_nodes + 1, dtype='int64')
    # copy the next indptr if current column is empty
    j_next = 0
    for (j, ind_ptr) in zip(unique_sorted_indices, column_slicing_indptr_temp):
        column_slicing_indptr[j_next:j + 1] = ind_ptr
        j_next = j + 1
    return column_slicing_permutation, column_slicing_indptr






def _augmented_admm(graph, num_classes=2, labels=None, eps_abs=1e-3, eps_rel=1e-3, tmax=10000, rho=0.1, eta=2, mu=10, verbosity=1, x0=0, y0=0, y1=0, return_y=False, return_x_list=False):
    num_nodes = graph.W.shape[0]




    x_list = []

    W = graph.W.tocsr()

    (column_slicing_permutation, column_slicing_indptr) = get_slicing_permutations(W)

    grad = lambda x: gradient(W, x, p=1)
    div = lambda z: divergence(W, z, column_slicing_permutation, column_slicing_indptr, p=1)

    def projection(x):
        x_ = simplex_projection(x + 1, axis=2) - 1
        return label_projection(x_, labels)

    Xtp1 = projection(x0 * np.ones((num_nodes, num_classes)))
    if return_x_list:
        x_list.append(Xtp1)
    Yt = y0 * np.ones((W.data.size, num_classes))
    if y1 is None:
        Ct = rho * grad(0 * Xtp1)
        Ctp1 = Yt + rho * grad(Xtp1)
        Yt = np.maximum(0, np.minimum(1, Ct))
        Ytp1 = np.maximum(0, np.minimum(1, Ctp1))
    else:
        Ytp1 = y1 * np.ones((W.data.size, num_classes))

    W2 = W.power(2)
    d = np.array(2 * np.sum(W2 + W2.T, 1))[:, 0]

    t = 0
    t_check_rho = 1
    i_check = 2

    cont = True
    while cont:
        t = t + 1
        Xt = Xtp1
        Ytm1 = Yt
        Yt = Ytp1
        Atp1 = -div(2 * Yt - Ytm1)
        rho_d = sps.diags(np.array(rho * d))
        rho_d_inv = sps.diags(1 / np.array(rho * d))
        Btp1 = rho_d_inv.dot(rho_d.dot(Xt) - Atp1)
        Xtp1 = projection(Btp1)
        if return_x:
            x_list.append(Xtp1)
        Ctp1 = Yt + rho * grad(Xtp1)
        Ytp1 = np.maximum(0, np.minimum(1, Ctp1))
        Rtp1 = Ytp1 - Yt  # actually this is Rtp1*rho
        Stp1 = div(rho * grad(Xtp1 - Xt) +
                   2 * Yt - Ytm1 - Ytp1)
        rtp1 = norm(Rtp1) / rho
        stp1 = norm(Stp1)
        ePri = sqrt(num_classes) * num_nodes * eps_abs + eps_rel * norm(grad(Xtp1))
        eDual = sqrt(num_classes * num_nodes) * eps_abs + eps_rel * \
                norm(div(Ytp1))
        if t >= t_check_rho:
            if rtp1 * eDual >= stp1 * ePri * mu:
                rho = rho * eta
            elif rtp1 * eDual <= stp1 * ePri / mu:
                rho = rho / eta
            t_check_rho = t_check_rho * i_check
        cont = (rtp1 > ePri or stp1 > eDual) and t < tmax
        if verbosity > 1 and t % 10 == 0:
            print("'\rK={K}, Keff={Keff}, {t:6d}: {rtp1:.2e}>{ePri:.2e} or {stp1:.2e}>{eDual:.2e}".format(
                K=num_classes, Keff=np.sum(np.any(Xtp1 > 0, 0)), t=t, rtp1=rtp1, ePri=ePri, stp1=stp1,
                eDual=eDual),
                end='')

    if verbosity > 0:
        print("\rK={K}, Keff={Keff}, {t}: TV={tv:.2f}, TVround={tvR:.2f}".format(
            K=num_classes, Keff=np.sum(np.any(Xtp1 > 0, 0)), t=t, tv=TV(W, Xtp1), tvR=TV(W, _roundSolution(Xtp1))))
    l_est = np.argmax(Xtp1, 1)
    X = Xtp1

    intermediate_results = {}
    if return_y:
        intermediate_results['y'] = (Yt, Ytp1)
    if return_x_list:
        intermediate_results['x_list'] = x_list

    return l_est, X, intermediate_results


class TvConvex(NodeLearner):
    def __init__(self, num_classes=2, verbosity=0, save_intermediate=False,eps_abs=1e-3, eps_rel=1e-3, tmax=10000, rho=0.1, eta=2,mu=10, verbosity=1, x0=0, y0=0, y1=0, **kwargs):
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        if (labels is None or len(labels['i']) == 0):
            i = np.argmax(np.sum(graph.W.maximum(0), 1))
            labels = {'i': [i], 'k': [0]}
            warnings.warn(
                'TVMinimization needs at least one label.\nI will continue with node {i} (highest positive degree) being a labelled node of cluster {k}'.format(
                    i=labels['i'], k=labels['k']))

        pass

    def cluster(graph, labels=None, , return_intermediate=False, return_y=False,
                return_x=False):