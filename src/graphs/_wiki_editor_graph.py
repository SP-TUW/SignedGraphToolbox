# the data files are from the paper
#
# @inproceedings{Yua17SNE,
#   author       = {Yuan, Shuhan and Wu, Xintao and Xiang, Yang},
#   booktitle    = {Pacific-Asia conference on knowledge discovery and data mining},
#   organization = {Springer},
#   pages        = {183--195},
#   title        = {SNE: signed network embedding},
#   year         = {2017}
# }
# available at
# https://bitbucket.org/bookcold/sne-signed-network-embedding/
import json

from ._graph import Graph
from src.tools.graph_tools import get_connected_components
from scipy.sparse import csr_matrix
import numpy as np
import os


class WikiEditorGraph(Graph):
    DATA_DIR = os.path.join('data', 'wiki_editor')

    def __init__(self, only_pos=False, **kwargs):
        l0, W = WikiEditorGraph.__get_labels_and_weights(only_pos=only_pos)
        super().__init__(W, l0, **kwargs)

    @staticmethod
    def __read_data():
        edge_file_name = os.path.join(WikiEditorGraph.DATA_DIR, 'wiki_edit.txt')
        label_file_name = os.path.join(WikiEditorGraph.DATA_DIR, 'wiki_usr_labels.txt')
        edge_data = np.genfromtxt(edge_file_name, delimiter='\t', dtype='int')
        label_data = np.genfromtxt(label_file_name, delimiter='\t', dtype='int')
        return edge_data, label_data

    @staticmethod
    def __preprocess_data(edge_data, label_data):
        edge_data_ = edge_data.copy()
        label_data_ = label_data.copy()

        i_sort_node_id = np.argsort(label_data_[:, 0])
        all_nodes_id = label_data_[i_sort_node_id, 0]
        id_to_ind = {node_id: i for i, node_id in enumerate(all_nodes_id)}
        for i in range(edge_data_.shape[0]):
            edge_data_[i, 0] = id_to_ind[edge_data_[i, 0]]
            edge_data_[i, 1] = id_to_ind[edge_data_[i, 1]]
        for i in range(label_data_.shape[0]):
            label_data_[i, 0] = id_to_ind[label_data_[i, 0]]

        # edges_data has weights in {0,1} -> map to {-1,1}
        edge_data_[:, 2] = (edge_data_[:, 2] * 2 - 1)
        label_data_ = label_data_[i_sort_node_id, :]
        return edge_data_, label_data_

    @staticmethod
    def __get_labels_and_weights(only_pos=False):
        preprocessed_data_file_name = os.path.join(WikiEditorGraph.DATA_DIR,'preprocessed_data.json')
        if os.path.exists(preprocessed_data_file_name):
            with open(preprocessed_data_file_name) as preprocessed_data_file:
                preprocessed_data = json.load(preprocessed_data_file)
            edge_data = np.array(preprocessed_data['edge_data'])
            label_data = np.array(preprocessed_data['label_data'])
        else:
            edge_data_orig, label_data_orig = WikiEditorGraph.__read_data()
            edge_data, label_data = WikiEditorGraph.__preprocess_data(edge_data_orig,label_data_orig)
            with open(preprocessed_data_file_name,'w') as preprocessed_data_file:
                preprocessed_data= {'edge_data': edge_data.tolist(), 'label_data': label_data.tolist()}
                json.dump(preprocessed_data, preprocessed_data_file)

        num_nodes = label_data.shape[0]

        labels = label_data[:, 1]

        # all edges are undirected and edge data contains only one entry for each edge
        # add both directions to the weight matrix
        i = np.r_[edge_data[:, 0],edge_data[:, 1]]
        j = np.r_[edge_data[:, 1],edge_data[:, 0]]
        v = np.r_[edge_data[:, 2],edge_data[:, 2]]
        if only_pos:
            v = np.maximum(v,0)
        weights = csr_matrix((v, (i, j)), shape=(num_nodes, num_nodes), dtype='float')
        weights.eliminate_zeros()

        ccs = get_connected_components(weights, use_csgraph=True)
        largest_cc = ccs[0]

        weights = weights[largest_cc, :]
        weights = weights[:, largest_cc]
        labels = labels[largest_cc]

        return labels, weights


if __name__ == '__main__':
    # run this module to generate the json representation of the voter array
    # this improves the runtime of subsequent runs significantly faster
    g = WikiEditorGraph(do_safe_voter_array=True)
