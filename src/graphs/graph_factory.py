import itertools
from enum import Enum

import numpy as np
import scipy.sparse as sps

from src.graphs import StochasticBlockModel as SBM
# from src.graphs.staticModels.UciGama import UciGama
# from src.graphs.staticModels.MultiCirculant import MultiCirculant
# from src.graphs.staticModels.SNAPNetwork import SNAPNetwork
# from src.graphs.staticModels.WikiEditorGraph import WikiEditorGraph
# from src.graphs.staticModels.WikiElecGraph import WikiElecGraph
# from src.graphs.staticModels.WikiRFAGraph import WikiRFAGraph
# from src.graphs.staticModels.KNNGraph import KNNGraph
# from src.graphs.staticModels.TVMinimizationGraph import TVMinimizationGraph
# from src.graphs.dynamicModels.DynSBM import DynSBM
# from src.graphs.Graph import Graph
from src.tools.simulation_tools import str_to_enum


def make_graph(model, num_largest_classes=None, **kwargs):
    # models = Enum('models', 'SBM TM UCI_GAMA MULTI_CIRC SNAP WIKI_EDITOR WIKI_ELEC WIKI_RFA KNN TVMinimization DYN_SBM')
    models = Enum('models', 'SBM')
    model_enum = str_to_enum(model, models)
    if model_enum is models.SBM:
        graph = SBM(**kwargs)
    # elif model_enum is models.UCI_GAMA:
    #     graph = UciGama(**kwargs)
    # elif model_enum is models.MULTI_CIRC:
    #     graph = MultiCirculant(**kwargs)
    # elif model_enum is models.SNAP:
    #     graph = SNAPNetwork(**kwargs)
    # elif model_enum is models.WIKI_EDITOR:
    #     graph = WikiEditorGraph(**kwargs)
    # elif model_enum is models.WIKI_ELEC:
    #     graph = WikiElecGraph(**kwargs)
    # elif model_enum is models.WIKI_RFA:
    #     graph = WikiRFAGraph(**kwargs)
    # elif model_enum is models.KNN:
    #     graph = KNNGraph(**kwargs)
    # elif model_enum is models.TVMinimization:
    #     graph = TVMinimizationGraph(**kwargs)
    # elif model_enum is models.DYN_SBM:
    #     graph = DynSBM(**kwargs)
    else:
        raise ValueError("{dn} not yet implemented".format(dn=model))

    if num_largest_classes is not None:
        graph.reduce_to_largest(num_largest_classes)

    return graph


def coarsen_graph(graph):
    # Graph coarsening according to Dhi07, Section 5.1
    W = graph.W.A
    num_nodes = graph.N
    node_to_supernode = np.zeros(num_nodes, dtype='int')
    unmarked_nodes = list(range(num_nodes))
    i_supernode = 0
    l0=[]
    while len(unmarked_nodes) > 0:
        # select one node
        ## index within unmarked nodes
        i_node = np.random.randint(0, len(unmarked_nodes))
        ## global index
        node = unmarked_nodes[i_node]
        node_to_supernode[node] = i_supernode
        l0.append(graph.l0[node])
        i_to_be_deleted = i_node

        # find unmarked neighbors
        neighbors = np.flatnonzero(W[node, unmarked_nodes] > 0)
        if len(neighbors) > 0:
            # find neighbors with highest weight
            neighbor_weights = W[node, neighbors]
            max_neighbors = np.flatnonzero(neighbor_weights==np.max(neighbor_weights))
            ## index within neighbors
            i_max_neighbor = np.random.choice(max_neighbors)
            ## index within unmarked nodes
            i_merge_neighbor = neighbors[i_max_neighbor]
            ## global index
            merge_neighbor = unmarked_nodes[i_merge_neighbor]
            node_to_supernode[merge_neighbor] = i_supernode
            i_to_be_deleted = max(i_node, i_merge_neighbor)
            del unmarked_nodes[i_to_be_deleted]
            i_to_be_deleted = min(i_node, i_merge_neighbor)
        del unmarked_nodes[i_to_be_deleted]
        i_supernode +=1

    num_supernodes = i_supernode
    w_coarse = np.zeros((num_supernodes,num_supernodes))
    for (i,j) in itertools.combinations(range(num_supernodes),2):
        i_joined = (node_to_supernode == i).nonzero()[0]
        j_joined = (node_to_supernode == j).nonzero()[0]
        w_coarse[i,j] = np.sum(W[np.ix_(i_joined,j_joined)])
    w_coarse += w_coarse.T

    coarse_graph = Graph(sps.csr_matrix(w_coarse), np.array(l0, dtype='int'))
    return coarse_graph, node_to_supernode


