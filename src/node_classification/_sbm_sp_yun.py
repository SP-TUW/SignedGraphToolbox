'''
Implementation of the spectral partitioning method of
    Yun, S. Y., & Proutiere, A. (2015).
    Optimal cluster recovery in the labeled stochastic block model.
    arXiv preprint arXiv:1510.05956. ISO 690
for the stimation of cluster association for the labeled stochastic block model (LSBM) with binary labels.
We cover the case where the numbe of clusters is known beforehand and therefore we will use the spectral decomposition from
    Yun, S. Y., & Proutiere, A. (2014, May).
    Community detection via random and adaptive sampling.
    In Conference on learning theory (pp. 138-175). PMLR.

'''

import numpy as np
from numpy.linalg import norm

from ._node_learner import NodeLearner


def _calc_log_label_probabilities(pi, pe, li, le, num_clusters=2):
    p0 = 1 - pi
    pp1 = pi * (1 - li)
    pm1 = pi * li
    q0 = 1 - pe
    qp1 = pe * (1 - le)
    qm1 = pe * le

    ep0 = (p0 - q0) * np.eye(num_clusters) + q0 * np.ones((num_clusters, num_clusters))
    epp1 = (pp1 - qp1) * np.eye(num_clusters) + qp1 * np.ones((num_clusters, num_clusters))
    epm1 = (pm1 - qm1) * np.eye(num_clusters) + qm1 * np.ones((num_clusters, num_clusters))
    label_probability = np.zeros((num_clusters, 3, num_clusters))
    label_probability[:, 0, :] = ep0
    label_probability[:, 1, :] = epp1
    label_probability[:, 2, :] = epm1
    return np.log(label_probability)


def _spectral_decomposition(a_gamma, est_avg_deg, num_clusters):
    n = a_gamma.shape[0]
    if num_clusters == 2:
        [d, v] = np.linalg.eig(a_gamma)
        x1 = v[:, 0]
        x2 = v[:, 1]
        if sum(x1) * sum(x2) > 0:
            x2 = -x2
        x = x1 + x2 - 1 / n * np.ones((n, n)).dot(x1 + x2)
        l = np.random.randint(0, 2, n)
        l[np.flatnonzero(x > 0)] = 0
        l[np.flatnonzero(x < 0)] = 1
    else:
        [d, v] = np.linalg.eig(a_gamma)
        a_hat = v[:,:num_clusters].T
        q = []
        t = []
        sum_t = []
        log_n = np.ceil(np.log(n)).astype('int')
        r = np.zeros(log_n)
        for i in range(log_n):
            q.append([])
            for v in range(n):
                q[i].append(set([w for w in range(n) if norm(a_hat[:,w]-a_hat[:,v])**2<=(i+1)*est_avg_deg/100]))
            t.append([])
            sum_t.append(set())
            xi_i = np.zeros((num_clusters,num_clusters))
            for k in range(num_clusters):
                v_k_max = np.argmax([len(q[i][v]-sum_t[i]) for v in range(n)])
                t[i].append(q[i][v_k_max]-sum_t[i])
                sum_t[i].update(t[i][k])
                xi_i[k,:] = np.mean(a_hat[:,list(t[i][k])],1)
            for v in list(set(range(n))-sum_t[i]):
                k = np.argmin([norm(a_hat[:,v]-xi_i[k,:]) for k in range(num_clusters)])
                t[i][k].add(v)
            r[i] = 0
            for k in range(num_clusters):
                r[i] = r[i] + norm(a_hat-xi_i[k,:][:,np.newaxis])**2

        i = np.argmin(r)
        l = np.zeros(n,dtype='int')
        for k in range(num_clusters):
            l[list(t[i][k])] = k




    c_assoc_gamma = np.zeros((n, num_clusters))
    c_assoc_gamma[range(n),l] = 1
    return c_assoc_gamma

class SbmSpYun(NodeLearner):
    def __init__(self, pi, pe, li, le, num_classes=2, class_distribution=None, verbosity=0, save_intermediate=False,
                 eps=1e-5, t_max=1e5):
        self.eps = eps
        self.t_max = t_max
        self.use_quadratic = False

        if class_distribution is None:
            self.class_distribution = np.ones(num_classes) / num_classes
        else:
            self.class_distribution = np.array(class_distribution) / sum(class_distribution)

        # in the paper they estimate this from the data in step 5
        self.log_label_prob = _calc_log_label_probabilities(pi, pe, li, le, num_classes)

        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)


    def estimate_labels(self, graph, labels=None, guess=None):
        if labels is not None:
            print('LsbmSpYun does not support labels')

        num_nodes = graph.num_nodes

        labels_matrix = graph.weights.A.astype('int')

        if guess is None:
            # line 1
            est_avg_deg = graph.weights.nnz / (num_nodes * (num_nodes - 1))

            # line 2
            rand_weights = np.random.rand(3)
            rand_weights[0] = 0
            rand_weights[0] = 1
            rand_weights[0] = 0
            A = rand_weights[labels_matrix]

            # line 3
            deg = np.sum(np.abs(labels_matrix), 1)
            deg_sorted = np.argsort(deg)
            num_removed_nodes = np.floor(num_nodes * np.exp(-num_nodes * est_avg_deg)).astype('int')
            if num_removed_nodes > 0:
                gamma = np.sort(deg_sorted[:-num_removed_nodes])
            else:
                gamma = np.arange(0, num_nodes)
            a_gamma = A
            a_gamma = a_gamma[gamma, :]
            a_gamma = a_gamma[:, gamma]

            # line 4 approximation of the algorithm by kmeans -- this is not exactly the same as in the paper
            # u,s,vh = np.linalg.svd(a_gamma)
            # s[num_clusters+1:] = 0
            # l = Kmeans.cluster(u[:,1:num_clusters], num_clusters)
            # c_assoc_gamma = np.zeros((num_nodes, num_clusters))
            # c_assoc_gamma[range(num_nodes),l] = 1
            c_assoc_gamma = _spectral_decomposition(a_gamma, est_avg_deg, self.num_classes)
            c_assoc = np.zeros((num_nodes, self.num_classes))
            for (i, j) in zip(gamma, range(len(gamma))):
                c_assoc[i, :] = c_assoc_gamma[j, :]
        else:
            l = guess
            c_assoc = np.zeros((num_nodes, self.num_classes))
            c_assoc[range(num_nodes), l] = 1



        # line 6
        for t in range(int(np.ceil(np.log(num_nodes)))):
            c_assoc_new = np.zeros((num_nodes, self.num_classes))
            for n in range(num_nodes):
                likelihood = np.sum(self.log_label_prob.dot(c_assoc.T)[:, labels_matrix[n, :], range(num_nodes)], 1)
                k = np.random.choice(np.flatnonzero(likelihood == likelihood.max()))
                c_assoc_new[n, k] = 1
            c_assoc = c_assoc_new

        l_est = np.argmax(c_assoc, 1)
        self.embedding = c_assoc
        return l_est
