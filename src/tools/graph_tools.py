from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as get_csgraph_cc


def get_gradient_matrix(weights, p, return_div=False):
    num_edges = weights.data.size
    num_nodes = weights.shape[0]
    weights_coo = weights.tocoo()
    i = np.arange(num_edges)
    i = np.tile(i, 2)
    j = np.r_[weights_coo.row, weights_coo.col]
    v = np.power(np.abs(weights_coo.data), 1 / p)
    v = np.concatenate((v, -np.sign(weights_coo.data) * v))
    gradient_matrix = csr_matrix((v, (i, j)), shape=(num_edges, num_nodes))
    if not return_div:
        return gradient_matrix
    else:
        divergence_matrix = -gradient_matrix.T
        return gradient_matrix, divergence_matrix


def select_labels(graph, label_amount, is_percentage=False, sorting_level=0):
    '''
    select a set of class labels based on the specified amount of labels.

    :param graph: An instance of src.graphs.Graph
    :param label_amount: Specifies how many labels there should be per class. Can be a scalar or an array with length graph.K0. In case a scalar is provided this number will be applied to each class; an array provides an individual value for each class. If this is a percentage, it has to be in [0,100]
    :param is_percentage: Specifies whether label_amount is a percentage or a number value.
    :param sorting_level: specifies how much sorting is carried out in the labels. This option is solely for better readability in debugging.
     - level 0: sort by class
     - level 1: sort by class first and sort by index afterwards
     - level 2: sort by index
    :return: dictionary of labeled nodes with node indices 'i' and corresponding class association 'k'. Both fields each contain a single python list.
    '''

    num_classes = graph.num_classes
    num_nodes = graph.num_nodes
    nodes_in_class = np.zeros((num_nodes,num_classes), dtype=bool)
    for k in range(num_classes):
        nodes_in_class[:,k] = graph.class_labels == k
    num_nodes_in_class = np.sum(nodes_in_class,axis=0)
    labels = {"i": np.array([],dtype=int), "k": []}
    if is_percentage:
        labels_per_class = np.round((num_nodes_in_class * label_amount) // 100).astype(dtype=int)
    else:
        labels_per_class = np.round(label_amount * np.ones(num_classes)).astype(dtype=int)
    print('{n} labels in class'.format(n=labels_per_class))
    for k in range(num_classes):
        if num_nodes_in_class[k] < labels_per_class[k]:
            raise ValueError("Class {k} should have {l} labels but it has only {n} nodes".format(k=k, l=labels_per_class[k], n=num_nodes_in_class[k]))
        labels_i = np.random.choice(np.flatnonzero(nodes_in_class[:, k]), labels_per_class[k], replace=False)
        if sorting_level == 1:
            labels_i = np.sort(labels_i)
        labels["i"] = np.r_[labels["i"], labels_i]
        labels["k"] += [k] * labels_per_class[k]
    if sorting_level == 2:
        i_sort = np.argsort(labels["i"])
        labels["i"] = labels["i"][i_sort]
        labels["k"] = np.array(labels['k'])[i_sort].tolist()
    labels["i"] = labels["i"].tolist()
    return labels


def get_connected_nodes(adjacency_matrix, node):
    num_nodes = adjacency_matrix.shape[0]
    visited = np.zeros(num_nodes, dtype='bool')
    nodes_to_visit = deque(node)
    visited[nodes_to_visit] = True
    while len(nodes_to_visit) > 0:
        node = nodes_to_visit.popleft()
        neighbors = adjacency_matrix.getrow(node).indices
        unvisited_neighbors = neighbors[np.bitwise_not(visited[neighbors])]
        visited[unvisited_neighbors] = True
        nodes_to_visit.extend(unvisited_neighbors)
    connected_nodes = np.flatnonzero(visited)
    return connected_nodes, visited


def get_connected_components(weight_matrix, use_csgraph=True):
    # build adjacency matrix (1 if there is a connection of any type in any direction between two nodes
    #                         0 else)
    num_nodes = weight_matrix.shape[0]
    i, j = np.nonzero(weight_matrix)
    ij = i.tolist() + j.tolist()
    ji = j.tolist() + i.tolist()
    adjacency_matrix = csr_matrix(([True] * len(ij), (ij, ji)), shape=(num_nodes, num_nodes))
    if use_csgraph:
        k, l = get_csgraph_cc(adjacency_matrix,connection='weak')
        ccs = []
        lengths = []
        for i in range(k):
            cc_mask_i = l==i
            ccs.append(np.flatnonzero(cc_mask_i))
            lengths.append(np.sum(cc_mask_i))
    else:
        unvisited = np.arange(num_nodes)
        ccs = []
        lengths = []
        while unvisited.size > 0:
            next_node = np.random.choice(unvisited, size=1)
            cc, in_cc = get_connected_nodes(adjacency_matrix, next_node)
            unvisited = unvisited[np.bitwise_not(in_cc[unvisited])]
            ccs.append(cc)
            lengths.append(cc.size)
    i_sort = np.argsort(lengths)
    return [ccs[i] for i in reversed(i_sort)]


def calc_signed_cut(W, l):
    w_pos = W.maximum(0)
    w_neg = -W.minimum(0)

    c = 0
    for k in range(np.max(l)+1):
        indicator = l==k
        ind_int = np.flatnonzero(indicator)
        ind_ext = np.flatnonzero(np.bitwise_not(indicator))
        w_pos_ext_triu = w_pos[ind_int,:]
        w_pos_ext_triu = w_pos_ext_triu[:,ind_ext]
        w_pos_ext_tril = w_pos[:,ind_int]
        w_pos_ext_tril = w_pos_ext_tril[ind_ext,:]
        w_neg_int = w_neg[ind_int,:]
        w_neg_int = w_neg_int[:,ind_int]
        c += w_pos_ext_triu.sum()
        # c += w_pos_ext_tril.sum()
        c += w_neg_int.sum()

    # w_full = W.A
    # w_pos = np.maximum(0, w_full)
    # weights_neg = np.maximum(0, -w_full)
    #
    # indicator = l[:, np.newaxis] == l[np.newaxis, :]
    # c_ = np.sum(weights_neg * indicator) + np.sum(w_pos * np.bitwise_not(indicator))
    return c