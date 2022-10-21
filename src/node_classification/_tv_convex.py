import warnings

import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from scipy import sparse as sps

from src.graphs import Graph
from src.node_classification._node_learner import NodeLearner
from src.tools.projections import simplex_projection, label_projection
from src.tools.graph_tools import calc_signed_cut


def _roundSolution(X):
    K = X.shape[1]
    ip = np.any(X > -1, 1)
    il = np.argmax(X[ip, :], 1)
    X_ = np.zeros(X.shape)
    X_[~ip, :] = 2 / K - 1
    X_[ip, :] = -1
    X_[ip, il] = 1
    return X_


def TV(W, X, p=1):
    return np.sum(np.maximum(_gradient(W, X, p=p), 0) ** p)


def _gradient(W, X, p=1):
    assert sps.isspmatrix_csr(W)
    num_nodes = W.shape[0]
    w_row_sliced = (np.sign(W.data) * np.abs(W.data) ** (1 / p))[:, np.newaxis]
    i = np.repeat(np.arange(num_nodes), np.diff(W.indptr))
    j = W.indices
    WaXi = X[i, :] * np.abs(w_row_sliced)
    WXj = X[j, :] * w_row_sliced
    return WaXi - WXj


def _divergence(W, Z, column_slicing_permutation, column_slicing_indptr, p=1):
    num_nodes = W.shape[0]
    num_clusters = Z.shape[1]
    w_row_sliced = (np.sign(W.data) * np.abs(W.data) ** (1 / p))[:, np.newaxis]
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


def _get_slicing_permutations(W):
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


def _get_regularizer(labels, is_sim_neighbor, reg_weights):
    '''
    turn list of regularization weights per labeled node into weight matrix
    :param labels: dict {'i': iterable, 'k': iterable} of label index and class association
    :param is_sim_neighbor: mask of sim_neighbors for all labelled nodes
    :param reg_weights: list of regularization weights
    :return:
    '''
    num_nodes = is_sim_neighbor.shape[0]
    sparse_reg_data = []
    reg_ij = []
    reg_ji = []
    for i, label in enumerate(labels):
        i_sim_neighbors = np.flatnonzero(is_sim_neighbor[:, i])
        n_sim_neighbors = i_sim_neighbors.size
        sparse_reg_data += [reg_weights[i]] * n_sim_neighbors
        reg_ij += [label] * n_sim_neighbors
        reg_ji += i_sim_neighbors.tolist()
    sparse_reg_data *= 2
    reg_ij, reg_ji = (reg_ij + reg_ji, reg_ji + reg_ij)
    regularizer = sps.csc_matrix((sparse_reg_data, (reg_ij, reg_ji)), shape=(num_nodes, num_nodes))
    return regularizer


def _run_rangapuram_resampling(graph, num_classes, labels, resampling_x_min, x0, y0, y1, save_intermediate, verbosity,
                               penalty_parameter, **kwargs):
    num_nodes = graph.num_nodes

    labels_i = np.array(labels['i'])
    labels_k = np.array(labels['k'])

    resampled_labels = {'i': labels['i'].copy(),
                        'k': labels['k'].copy()}

    is_label = np.zeros(num_nodes, dtype=bool)
    label_encoding = np.zeros((num_nodes, num_classes))

    is_label[labels['i']] = True
    label_encoding[is_label, :] = -1
    label_encoding[is_label, labels['k']] = 1

    X = x0
    Yt = y0
    Ytp1 = y1

    num_resampled = np.sum(labels_k[:,None]==np.arange(num_classes)[None,:],axis=0)

    converged = False
    while not converged:
        l_est, X, (Yt, Ytp1, penalty_parameter), intermediate_results = _run_augmented_admm(graph=graph,
                                                                                            num_classes=num_classes,
                                                                                            labels=resampled_labels,
                                                                                            verbosity=verbosity,
                                                                                            return_state=True, x0=X,
                                                                                            y0=Yt, y1=Ytp1,
                                                                                            penalty_parameter=penalty_parameter,
                                                                                            save_intermediate=save_intermediate,
                                                                                            **kwargs)
        is_degenerate = np.all(X <= resampling_x_min, axis=1)
        num_degenerate = np.sum(is_degenerate)
        if num_degenerate>0:
            if verbosity>=1:
                print('{n} degenerate entries left'.format(n=num_degenerate))
            l_est_rand_resolve = l_est.copy()
            is_tied = np.diff(np.sort(X,axis=1)[:,-2:]).squeeze() == 0
            num_tied = np.sum(is_tied)
            l_est_rand_resolve[is_tied] = np.random.randint(0,num_classes, num_tied)
            x_switch = -np.ones((num_nodes,num_classes))
            x_switch[range(num_nodes),l_est_rand_resolve] = 1

            class_size = np.sum(l_est_rand_resolve[:,None]==np.arange(num_classes)[None,:],axis=0)
            num_resampled = np.maximum(np.minimum(class_size, 2*num_resampled),1)
            rankings = np.zeros((num_nodes,num_classes),dtype=int)

            resampled_labels['i'] = []
            resampled_labels['k'] = []

            sc = calc_signed_cut(graph.weights, l_est_rand_resolve)

            for k in range(num_classes):
                switching_cost = np.zeros(num_nodes)
                class_k = np.flatnonzero(l_est_rand_resolve==k)
                for i in class_k:
                    switching_cost[i] = float('inf')
                    if not is_label[i]:
                        w_pos_neighbors = graph.w_pos.getrow(i)
                        w_neg_neighbors = graph.w_neg.getrow(i)
                        pos_neighbors = w_pos_neighbors.indices
                        neg_neighbors = w_neg_neighbors.indices
                        pos_cut = 2 * w_pos_neighbors.data[None, :].dot(l_est_rand_resolve[pos_neighbors][:, None] != k)
                        neg_cut = 2 * w_neg_neighbors.data[None, :].dot(l_est_rand_resolve[neg_neighbors][:, None] == k)
                        for l in range(num_classes):
                            if l != k:
                                # test switching node i from k to l
                                pos_cut_switch = 2 * w_pos_neighbors.data[None, :].dot(l_est_rand_resolve[pos_neighbors][:, None] != l)
                                neg_cut_switch = 2 * w_neg_neighbors.data[None, :].dot(l_est_rand_resolve[neg_neighbors][:, None] == l)
                                sc_switch = sc-pos_cut-neg_cut+pos_cut_switch+neg_cut_switch
                                # x_switch[i,:] = -1
                                # x_switch[i,l] = 1
                                # tv = TV(graph.weights,x_switch,p=1)
                                # l_switch = l_est_rand_resolve.copy()
                                # l_switch[i] = l
                                # sc_test = calc_signed_cut(graph.weights, l_switch)
                                if sc_switch < switching_cost[i]:
                                    switching_cost[i] = sc_switch
                        # reset node i to k
                        # x_switch[i,:] = -1
                        # x_switch[i,k] = 1
                rankings[:,k] = np.argsort(switching_cost)[::-1]
                resampled_labels['i'] += rankings[:num_resampled[k],k].tolist()
                resampled_labels['k'] += [k] * num_resampled[k]
        else:
            converged = True



    return l_est, X, intermediate_results


def _run_regularization(graph, num_classes, labels, verbosity, regularization_x_min, regularization_parameter,
                        regularization_max, x0, y0, y1, penalty_parameter, return_min_tv, save_intermediate, **kwargs):
    # enforce that label-neighbor-nodes with positive edges end up in the same cluster
    #                                        negative edges end up in different clusters
    # -> the solution where the nodes from the same cluster (for clusters 1...K-1) are clustered correctly and nodes that do not belong to any cluster end up in the K-th cluster should be the cheapest
    num_nodes = graph.num_nodes

    if labels is None:
        i = np.argmin(np.sum(graph.W.minimum(0), 1))
        labels = {'i': [i], 'k': [0]}
        warnings.warn(
            'TVMinimization needs at least one label.\nI will continue with node {i} (highest negative degree) being a labelled node of cluster {k}'.format(
                i=labels['i'], k=labels['k']))

    reg_weights = np.zeros(len(labels['i']))

    l_est_list = []

    # label_is_in_cluster = np.zeros((len(labels['i']), num_classes),dtype='bool')

    is_sim_neighbor = np.zeros((num_nodes, len(labels['i'])), dtype='bool')
    sim_weight = np.zeros((num_nodes, len(labels['i'])))
    num_cluster_sim_neighbor = np.zeros((num_nodes, num_classes), dtype='int')
    cluster_sim_weight = np.zeros((num_nodes, num_classes))
    for iL, (i, k) in enumerate(zip(labels['i'], labels['k'])):
        # neighbors with positive edges (.A turns sparse result into full array)
        is_sim_neighbor[:, iL] = (graph.weights[i, :] > 0).A[0, :]
        sim_weight[:, iL] = np.maximum(graph.weights[i, :].A, 0)
        is_sim_neighbor[labels['i'], iL] = False
        num_cluster_sim_neighbor[:, k] += is_sim_neighbor[:, iL]
        cluster_sim_weight[:, k] += sim_weight[:, iL]
        # label_is_in_cluster[iL, k] = True

    is_multi_sim_neighbor = np.sum(num_cluster_sim_neighbor > 0, axis=1) > 1
    for i in np.flatnonzero(is_multi_sim_neighbor):
        k_max = np.argmax(cluster_sim_weight[i, :])
        sim_weight_i = sim_weight[i, :]
        sim_weight_i[labels['k'] != k_max] = 0
        i_max = np.flatnonzero(sim_weight_i == np.max(sim_weight_i))
        is_sim_neighbor[i, :] = False
        is_sim_neighbor[i, i_max] = True
        # is_sim_neighbor[i,np.bitwise_not(label_is_in_cluster[:,k_max])] = False

    X = x0
    Yt = y0
    Ytp1 = y1
    weights = graph.weights.copy()
    doRegularize = True
    tv_min = float('inf')
    x_min = None
    l_est_min = None
    reg_weights_min = None
    while doRegularize:
        doRegularize = False
        reg_graph = Graph(num_classes=num_classes, class_labels=graph.class_labels, weights=weights)
        l_est, X, (Yt, Ytp1, penalty_parameter), intermediate_results = _run_augmented_admm(reg_graph,
                                                                                            num_classes=num_classes,
                                                                                            labels=labels,
                                                                                            verbosity=verbosity,
                                                                                            return_state=True, x0=X,
                                                                                            y0=Yt, y1=Ytp1,
                                                                                            penalty_parameter=penalty_parameter,
                                                                                            save_intermediate=save_intermediate,
                                                                                            **kwargs)
        l_est_list.append(l_est)
        x_est = -np.ones((num_nodes, num_classes))
        x_est[np.arange(num_nodes), l_est] = 1
        tv_est = TV(graph.weights, x_est, 1)
        if tv_est < tv_min:
            tv_min = tv_est
            x_min = X.copy()
            l_est_min = l_est.copy()
            reg_weights_min = reg_weights.copy()
        x_neighbor_min = float('inf')
        for (iL, i, k) in zip(range(len(labels['i']) + 1), labels['i'], labels['k']):
            if np.any(is_sim_neighbor[:, iL]):
                isPosSimNeighbor = np.logical_and(is_sim_neighbor[:, iL], X[:, k] > 0)  # 2/num_classes-1-1e-3)
                x_neighbor = np.min(X[:, k][isPosSimNeighbor]) if np.any(isPosSimNeighbor) else float('inf')
                x_neighbor_min = min(x_neighbor_min, x_neighbor)
                if not np.any(isPosSimNeighbor) or np.any(np.min(X[:, k][isPosSimNeighbor]) < regularization_x_min):
                    doRegularize = True
                    reg_weights[iL] = max(1, reg_weights[iL] * regularization_parameter)
        weights = graph.weights + graph.weights.multiply(_get_regularizer(labels['i'], is_sim_neighbor, reg_weights))
        if np.any(reg_weights > regularization_max):
            print("stopped due to maximum regularization")
            doRegularize = False
        if doRegularize and verbosity > 2:
            print("reg_weights = {reg_weights}".format(reg_weights=reg_weights))
        elif doRegularize and verbosity > 1:
            print("reg_weights max = {max_reg_weights}, x_neighbor_min = {x_neighbor_min}".format(
                max_reg_weights=np.max(reg_weights), x_neighbor_min=x_neighbor_min))

    if return_min_tv:
        intermediate_results['reg_weights'] = reg_weights_min
        return l_est_min, x_min, intermediate_results
    else:
        intermediate_results['reg_weights'] = reg_weights
        return l_est, X, intermediate_results


def _run_augmented_admm(graph, num_classes, labels, eps_abs, eps_rel, t_max, penalty_parameter, penalty_scaling,
                        penalty_threshold, verbosity, x0, y0, y1, return_state, save_intermediate):
    num_nodes = graph.weights.shape[0]

    intermediate_results = {}
    save_x_list = False
    save_y = False

    if save_intermediate is not None:
        if type(save_intermediate) is bool:
            if save_intermediate:
                save_x_list = True
                save_y = True
        elif type(save_intermediate) is list:
            if 'x_list' in save_intermediate:
                save_x_list = True
                intermediate_results['x_list'] = []

            if 'y' in save_intermediate:
                save_y = True
                intermediate_results['y'] = None
        else:
            raise ValueError(
                'don''t know how to interpret save_intermediate ({s}). Its value can either be bool or a list of strings'.format(
                    s=save_intermediate))

    W = graph.weights.tocsr()
    W2 = W.power(2)
    d = np.array(2 * np.sum(W2 + W2.T, 1))[:, 0]
    (column_slicing_permutation, column_slicing_indptr) = _get_slicing_permutations(W)

    grad_matrix, div_matrix = graph.get_gradient_matrix(p=1, return_div=True)

    grad = lambda x: grad_matrix.dot(x)#_gradient(W, x, p=1)
    div = lambda z: div_matrix.dot(z)#_divergence(W, z, column_slicing_permutation, column_slicing_indptr, p=1)
    projection = lambda x: label_projection(simplex_projection(x + 1, a=2, axis=1) - 1, labels)

    Xtp1 = projection(x0 * np.ones((num_nodes, num_classes)))
    if save_x_list:
        intermediate_results['x_list'].append(Xtp1)
    if y0 is None:
        y0 = 1
    Yt = y0 * np.ones((W.data.size, num_classes))
    if y1 is None:
        Ct = penalty_parameter * grad(0 * Xtp1)
        Ctp1 = Yt + penalty_parameter * grad(Xtp1)
        Yt = np.maximum(0, np.minimum(1, Ct))
        Ytp1 = np.maximum(0, np.minimum(1, Ctp1))
    else:
        Ytp1 = y1 * np.ones((W.data.size, num_classes))

    t = 0
    t_check_rho = 1
    i_check = 2

    converged = False
    while not converged:
        # update variables
        t = t + 1
        Xt = Xtp1
        Ytm1 = Yt
        Yt = Ytp1

        # update steps
        rho_d = sps.diags(np.array(penalty_parameter * d))
        rho_d_inv = sps.diags(np.array([1 / (penalty_parameter * di) if di != 0 else 0 for di in d]))

        # x
        Atp1 = -div(2 * Yt - Ytm1)
        Btp1 = rho_d_inv.dot(rho_d.dot(Xt) - Atp1)
        Xtp1 = projection(Btp1)

        # y
        Ctp1 = Yt + penalty_parameter * grad(Xtp1)
        Ytp1 = np.maximum(0, np.minimum(1, Ctp1))

        # residuals
        Rtp1 = Ytp1 - Yt  # actually this is Rtp1*penalty_parameter
        Stp1 = div(penalty_parameter * grad(Xtp1 - Xt) +
                   2 * Yt - Ytm1 - Ytp1)
        rtp1 = norm(Rtp1) / penalty_parameter
        stp1 = norm(Stp1)

        # tolerances
        ePri = sqrt(num_classes) * num_nodes * eps_abs + eps_rel * norm(grad(Xtp1))
        eDual = sqrt(num_classes * num_nodes) * eps_abs + eps_rel * \
                norm(div(Ytp1))

        # update penalty
        if t >= t_check_rho:
            if rtp1 * eDual >= stp1 * ePri * penalty_threshold:
                penalty_parameter = penalty_parameter * penalty_scaling
            elif rtp1 * eDual <= stp1 * ePri / penalty_threshold:
                penalty_parameter = penalty_parameter / penalty_scaling
            t_check_rho = t_check_rho * i_check

        # store intermediate result
        if save_x_list:
            intermediate_results['x_list'].append(Xtp1)

        if verbosity > 1 and t % 1 == 0:
            print("'\rK={K}, Keff={Keff}, {t:6d}: {rtp1:.2e}>{ePri:.2e} or {stp1:.2e}>{eDual:.2e}".format(
                K=num_classes, Keff=np.sum(np.any(Xtp1 > 0, 0)), t=t, rtp1=rtp1, ePri=ePri, stp1=stp1,
                eDual=eDual),
                end='')

        converged = (rtp1 <= ePri and stp1 <= eDual)
        if not converged and t > t_max:
            break

    if verbosity > 0:
        print("\rK={K}, Keff={Keff}, {t}: TV={tv:.2f}, TVround={tvR:.2f}".format(
            K=num_classes, Keff=np.sum(np.any(Xtp1 > 0, 0)), t=t, tv=TV(W, Xtp1), tvR=TV(W, _roundSolution(Xtp1))))
    l_est = np.argmax(Xtp1, 1)
    X = Xtp1

    if save_y:
        intermediate_results['y'] = (Yt, Ytp1)

    if return_state:
        return l_est, X, (Yt, Ytp1, penalty_parameter), intermediate_results
    else:
        return l_est, X, intermediate_results


class TvConvex(NodeLearner):
    def __init__(self, num_classes=2, verbosity=0, save_intermediate=None, eps_abs=1e-3, eps_rel=1e-3, t_max=10000,
                 penalty_parameter=0.1, penalty_scaling=2, penalty_threshold=10, degenerate_heuristic=None,
                 regularization_x_min=0.9, regularization_parameter=2, regularization_max=1024, return_min_tv=False,
                 resampling_x_min=0.1):
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.t_max = t_max
        self.penalty_parameter = penalty_parameter
        self.penalty_scaling = penalty_scaling
        self.penalty_threshold = penalty_threshold
        if degenerate_heuristic is not None and degenerate_heuristic not in ['regularize', 'rangapuram_resampling']:
            raise ValueError(
                'unknown heuristic ''{h}'' for degenerate solutions\nEither use None or ''regularize'' or ''rangapuram_resampling'''.format(
                    h=self.degenerate_heuristic))
        else:
            self.degenerate_heuristic = degenerate_heuristic
        self.regularization_x_min = regularization_x_min
        self.regularization_parameter = regularization_parameter
        self.regularization_max = regularization_max
        self.return_min_tv = return_min_tv
        self.resampling_x_min = resampling_x_min
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        if (labels is None or len(labels['i']) == 0):
            i = np.argmax(np.sum(graph.weights.maximum(0), 1))
            labels = {'i': [i], 'k': [0]}
            warnings.warn(
                'TVMinimization needs at least one label.\nI will continue with node {i} (highest positive degree) being a labelled node of cluster {k}'.format(
                    i=labels['i'], k=labels['k']))

        if type(guess) is dict:
            if 'x0' in guess:
                x0 = guess['x0']
            else:
                x0 = None
            if 'y0' in guess:
                y0 = guess['y0']
            else:
                y0 = None
            if 'y1' in guess:
                y1 = guess['y1']
            else:
                y1 = None
        else:
            x0 = np.ones((graph.num_nodes, self.num_classes))
            x0[range(graph.num_nodes), guess] = 1
            y0 = None
            y1 = None

        if self.degenerate_heuristic == 'regularize':
            l_est, X, intermediate_results = _run_regularization(graph=graph, num_classes=self.num_classes,
                                                                 labels=labels,
                                                                 regularization_x_min=self.regularization_x_min,
                                                                 regularization_parameter=self.regularization_parameter,
                                                                 regularization_max=self.regularization_max,
                                                                 return_min_tv=self.return_min_tv, eps_abs=self.eps_abs,
                                                                 eps_rel=self.eps_rel, t_max=self.t_max,
                                                                 penalty_parameter=self.penalty_parameter,
                                                                 penalty_scaling=self.penalty_scaling,
                                                                 penalty_threshold=self.penalty_threshold,
                                                                 verbosity=self.verbosity, x0=x0, y0=y0, y1=y1,
                                                                 save_intermediate=self.save_intermediate)
        elif self.degenerate_heuristic == 'rangapuram_resampling':
            l_est, X, intermediate_results = _run_rangapuram_resampling(graph=graph, num_classes=self.num_classes,
                                                                        labels=labels,
                                                                        resampling_x_min=self.resampling_x_min,
                                                                        eps_abs=self.eps_abs, eps_rel=self.eps_rel,
                                                                        t_max=self.t_max,
                                                                        penalty_parameter=self.penalty_parameter,
                                                                        penalty_scaling=self.penalty_scaling,
                                                                        penalty_threshold=self.penalty_threshold,
                                                                        verbosity=self.verbosity, x0=x0, y0=y0, y1=y1,
                                                                        save_intermediate=self.save_intermediate)
        elif self.degenerate_heuristic is None:
            l_est, X, intermediate_results = _run_augmented_admm(graph=graph, num_classes=self.num_classes,
                                                                 labels=labels, eps_abs=self.eps_abs,
                                                                 eps_rel=self.eps_rel, t_max=self.t_max,
                                                                 penalty_parameter=self.penalty_parameter,
                                                                 penalty_scaling=self.penalty_scaling,
                                                                 penalty_threshold=self.penalty_threshold,
                                                                 verbosity=self.verbosity, x0=x0, y0=y0, y1=y1,
                                                                 return_state=False,
                                                                 save_intermediate=self.save_intermediate)
        else:
            raise ValueError('unknown heuristic ''{h}'' for degenerate solutions'.format(h=self.degenerate_heuristic))

        is_degenerate = np.all(X <= self.resampling_x_min, axis=1)
        num_degenerate = np.sum(is_degenerate)
        print('{n} degenerate entries left'.format(n=num_degenerate))


        self.embedding = X
        self.normalized_embedding = (X + 1) / 2
        self.intermediate_results = intermediate_results

        return l_est
