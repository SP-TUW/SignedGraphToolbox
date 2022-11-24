import warnings

import numpy as np
from numpy import random
from scipy import sparse as sps

from ._graph import Graph


class StochasticBlockModel(Graph):
    '''
    This class is an implementation of the signed stochastic block model following the definition of the labelled
    stochastic block model from
        Heimlicher, S., Lelarge, M., & Massoulié, L. (2012).
        Community detection in the labelled stochastic block model.
        arXiv preprint arXiv:1209.2910.

    At the moment we support two types of label distributions:
        - Bernoulli ... labels/weights follow a Bernoulli distribution over {-1,1}
        - Gaussian ... labels/weights follow a Gaussian distribution
    '''

    def __init__(self, type, class_distribution, pi, pe, l0=None, num_nodes=None, random_classes=False, li=None,
                 le=None,
                 mui=None, sigmai=None, mue=None, sigmae=None, **kwargs):
        '''
        :param type: type of the label distribution
        :param class_distribution: Proportion of nodes that are in each class. If num_nodes is specified this will be rescaled such that it sums to one. Otherwise this will be treated as a number of nodes
        :param num_nodes: Total number of nodes. If not specified, this will be set to the sum over class_distribution
        :param random_classes: If True, the class sizes will be defined by a multinomial distribution. If False the class sizes will be defined deterministically via num_nodes and class_distribution
        :param pi: Probability that internal edges exist
        :param pe: Probability that external edges exist
        :param li: Probability that the weight of an internal edge is -1 in the Bernoulli model
        :param le: Probability that the weight of an external edge is -1 in the Bernoulli model
        :param mui: Mean of weights of internal edges in the Gaussian model
        :param sigmai: Standard deviation of weights of internal edges in the Gaussian model
        :param mue: Mean of weights of internal edges in the Gaussian model
        :param sigmae: Standard deviation of weights of internal edges in the Gaussian model
        :param kwargs:
        '''
        if type == 'Bernoulli':
            assert li is not None and le is not None, "li and le need to be specified for Bernoulli type"
            generate_weights = lambda nnz, is_intern: StochasticBlockModel.bernoulli_weights(nnz, li, le, is_intern)
        elif type == 'Gaussian':
            assert mui is not None and sigmai is not None and mue is not None and sigmae is not None, "li and le need to be specified for Bernoulli type"
            generate_weights = lambda nnz, is_intern: StochasticBlockModel.gaussian_weights(nnz, mui, sigmai, mue,
                                                                                            sigmae, is_intern)
        else:
            raise ValueError("unnknown type {type}".format(type=type))

        if l0 is None:
            num_classes = len(class_distribution)

            # parse input for number of nodes in each class
            if num_nodes is None:
                num_nodes = int(np.sum(class_distribution))
                num_nodes_in_class = np.round(class_distribution).astype('int')
                class_distribution = num_nodes_in_class / num_nodes
                if np.any(num_nodes_in_class - class_distribution != 0):
                    raise ValueError('class_distribution needs to be integer if num_nodes is not specified')
            else:
                class_distribution = np.array(class_distribution) / sum(class_distribution)
                num_nodes_in_class = np.round(num_nodes * class_distribution).astype('int')
                if not random_classes and np.sum(num_nodes_in_class) != num_nodes:
                    num_nodes = int(np.sum(num_nodes_in_class))
                    warnings.warn('for deterministic class assignment the class distribution does not allow the '
                                  'specified number of nodes. I will continue with num_nodes={nn}'.format(nn=num_nodes))

            if random_classes:
                num_nodes_in_class = np.random.multinomial(num_nodes, class_distribution)
        else:
            num_classes = np.max(l0) + 1
            num_nodes_in_class = np.sum(l0[:, np.newaxis] == np.arange(num_classes)[np.newaxis, :], axis=0)

        cumsum_nodes_in_class = np.r_[0, np.cumsum(num_nodes_in_class)]

        nodes_in_class = []
        if l0 is None:
            l0 = np.zeros(num_nodes,dtype=int)
            for k in range(num_classes):
                l0[cumsum_nodes_in_class[k]:cumsum_nodes_in_class[k + 1]] = k
                nodes_in_class.append(np.arange(cumsum_nodes_in_class[k],cumsum_nodes_in_class[k + 1]))
        else:
            for k in range(num_classes):
                nodes_in_class.append(np.flatnonzero(l0==k))

        # get number of existing edges for each pair of classes
        num_edges = np.zeros((num_classes, num_classes), dtype=int)
        for k1 in range(num_classes):
            for k2 in range(k1, num_classes):
                N2 = num_nodes_in_class[k1] * num_nodes_in_class[k2]
                if k1 == k2:
                    num_edges[k1, k2] = random.binomial(n=N2, p=pi, size=1)
                else:
                    num_edges[k1, k2] = random.binomial(n=N2, p=pe, size=1)

        # define the actual edges
        i = np.zeros(np.sum(num_edges), dtype=int)
        j = np.zeros(np.sum(num_edges), dtype=int)
        data = np.zeros(np.sum(num_edges))
        next_block = 0
        for k1 in range(num_classes):
            for k2 in range(k1, num_classes):
                N2 = num_nodes_in_class[k1] * num_nodes_in_class[k2]
                # draw linear indices of nonzero elements
                lin_ind = random.choice(N2, size=num_edges[k1, k2], replace=False)
                # convert to 2-D indices inside the classes
                i_ = nodes_in_class[k1][lin_ind % num_nodes_in_class[k1]]
                j_ = nodes_in_class[k2][lin_ind // num_nodes_in_class[k1]]
                # generate weights
                data_ = generate_weights(num_edges[k1, k2], k1 == k2)
                # store in big list
                i[next_block:next_block + num_edges[k1, k2]] = i_
                j[next_block:next_block + num_edges[k1, k2]] = j_
                data[next_block:next_block + num_edges[k1, k2]] = data_
                next_block = next_block + num_edges[k1, k2]
        data[i >= j] = 0
        i_list = i.tolist()
        j_list = j.tolist()
        data_list = data.tolist()
        ij = i_list + j_list
        ji = j_list + i_list
        dd = data_list + data_list
        W = sps.coo_matrix((dd, (ij, ji)), shape=(num_nodes, num_nodes))
        super().__init__(num_classes=num_classes,class_labels=l0,weights=W, **kwargs)

    @staticmethod
    def bernoulli_weights(nnz, li, le, is_intern):
        if is_intern:
            p_flip = li
        else:
            p_flip = le
        data = random.choice([-1, 1], size=nnz, p=[p_flip, 1 - p_flip], replace=True)
        return data

    @staticmethod
    def gaussian_weights(nnz, mui, sigmai, mue, sigmae, is_intern):
        if is_intern:  # inside class
            mu_ = mui
            sigma_ = sigmai

        else:  # outside class
            mu_ = mue
            sigma_ = sigmae
        data = mu_ + sigma_ * random.standard_normal(nnz)
        return data


if __name__ == '__main__':
    import src.graphs.factory as factory

    graph_config = {'model': 'SBM',
                    'type': 'Bernoulli',
                    'class_distribution': [1, 1, 1],
                    'num_nodes': 100,
                    'pi': 0.1,
                    'pe': 0.1,
                    'li': 0,
                    'le': 1}

    graph = factory.make_graph(**graph_config)
