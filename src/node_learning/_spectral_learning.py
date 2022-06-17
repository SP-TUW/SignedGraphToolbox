from scipy.sparse import csr_matrix, eye, issparse
from scipy.sparse.linalg import spsolve, eigsh
import numpy as np
from numpy import sqrt
from src.tools.projections import unitarization
from src.metric_learning import seeded_kmeans
from sklearn import KMeans


def _get_objective_matrices_and_eig_selector(graph, objective, num_clusters):

    if objective == "RC":
        d = abs(weight_matrix).sum(axis=1).T
        d_nz = np.where(d == 0, 1, d)
        deg_matrix = diags(data=d_nz, diags=0, m=graph.N, n=graph.N)
        a = deg_matrix - weight_matrix
        normalization = np.ones((graph.N, 1))
        eig_sel = 'SM'
    elif objective == "NC":
        d = abs(weight_matrix).sum(axis=1).T
        d_nz = np.where(d == 0, 1, d)
        deg_matrix = diags(data=d_nz, diags=0, m=graph.N, n=graph.N)
        a = deg_matrix - weight_matrix
        dd = sqrt(d)
        dd = np.where(dd == 0, 1, dd)
        inv_sqrt_deg_matrix = diags(data=1 / dd, diags=0, m=graph.N, n=graph.N)
        a = inv_sqrt_deg_matrix.dot(a.dot(inv_sqrt_deg_matrix))

        normalization = np.array(1 / dd).T
        eig_sel = 'SM'
    elif objective == "BNC":
        inv_sqrt_sig_deg = 1 / sqrt(abs(weight_matrix).sum(axis=1)).T
        inv_sqrt_sig_deg_matrix = diags(data=inv_sqrt_sig_deg, diags=0, m=graph.N, n=graph.N)
        neg_weight_matrix = -weight_matrix.minimum(0)
        neg_deg = neg_weight_matrix.sum(axis=1).T
        neg_deg_matrix = diags(data=neg_deg, diags=0, m=graph.N, n=graph.N)
        a = inv_sqrt_sig_deg_matrix.dot((neg_deg_matrix + weight_matrix)).dot(inv_sqrt_sig_deg_matrix)
        normalization = np.ones((graph.N, 1))
        eig_sel = 'LM'

    return a, eig_sel, normalization, force_unsigned


def _labels_to_lin_const(labels, num_nodes, num_clusters):
    num_labels = len(labels['i'])

    labels_i = np.array(labels['i'])
    labels_k = np.array(labels['k'])
    B = csr_matrix((np.ones(num_labels), (np.arange(num_labels), labels_i)), shape=(num_labels, num_nodes))
    c = np.zeros((num_labels, num_clusters))
    c[np.arange(num_labels), labels_k] = 1
    c = unitarization(c) * np.sqrt(num_labels / num_nodes)

    return B, c


def _joint_multiclass(self, obj_matrix, B, c, random_init, return_intermediate, eps=1e-5, t_max=1e5):
    '''
    This function implements our joint multiclass algorithm where the linear constraints are given by the labels

    In the case of c.shape[1]==1 it is equivalent to the method from :cite:p:`Xu09LinConstCut`

    :param obj_matrix: matrix for quadratic objective. Needs to be positive semi-definite
    :param B, c: the linear constraints on X of the form BX=c
    :return: vector v which maximizes the optimization problem.
    '''

    num_labels = c.shape[0]
    if len(c.shape) > 1:
        num_clusters = c.shape[1]
    else:
        num_clusters = 1
    num_nodes = obj_matrix.shape[0]

    obj_list = []

    # The algorithm says
    # P = I-B^T(BB^T)^{-1}B
    # but for our case BB^T=I and so we can leave this out
    if issparse(B):
        P = eye(num_nodes) - B.T.dot(spsolve(B.dot(B.T), B))
    else:
        P = np.eye(num_nodes) - B.T.dot(np.linalg.solve(B.dot(B.T), B))
    n0 = B.T.dot(c)
    if random_init:
        v0 = np.random.randn(n0.shape[0], n0.shape[1])
    else:
        v0 = n0
    gamma = np.sqrt(1 - np.linalg.norm(n0, axis=0) ** 2)
    v = v0

    t = 0
    converged = False
    while not converged:
        v_old = v.copy()
        PAv = P.dot(obj_matrix.dot(v_old))
        PAv_unit = unitarization(PAv)
        u = gamma * PAv_unit
        v = u + n0
        obj_list.append(np.trace(v.T.dot(obj_matrix.dot(v))))
        t += 1
        converged = np.linalg.norm(v - v_old) < eps * sqrt((num_nodes - num_labels) * num_clusters)
        if not converged and t >= t_max:
            break

    if return_intermediate:
        return v, obj_list, obj_matrix.copy()
    else:
        return v


def _sequential_multiclass(obj_matrix, B, c, random_init,
                           return_intermediate, eps, t_max):
    num_clusters = c.shape[1]
    num_nodes = obj_matrix.shape[0]

    obj_list = []

    v = np.zeros((num_nodes, num_clusters))
    for k in range(num_clusters):
        if k > 0:
            B_ = np.vstack([B.A, v[:, :k].T])
            c_ = np.concatenate((c[:, k], np.zeros((k))), axis=0)[:, None]
        else:
            B_ = B.A.copy()
            c_ = c[:, k][:, None]

        r_val = _joint_multiclass(obj_matrix, B_, c_, random_init=random_init, return_intermediate=return_intermediate,
                                 eps=eps, t_max=t_max)

        if return_intermediate:
            v[:, k] = r_val[0]
            obj_list.append(r_val[1])
        else:
            v[:, k] = r_val[:, 0]

    if return_intermediate:
        return v, obj_list, obj_matrix.copy()
    else:
        return v


class spectral_learning:

    def __init__(self, n_classes=2, objective='RC', multiclass_method='joint',
                random_init=False, verbose=0, eps=1e-5, t_max=1e5, save_intermediate=False):
        self.n_classes = n_classes
        self.objective = objective
        self.multiclass_method=multiclass_method
        self.random_init=random_init
        self.verbose = verbose
        self.eps=eps
        self.t_max = t_max
        self.save_intermediate = save_intermediate
        self.intermediate_results = None

    def estimate_labels(self, graph, labels=None):
        num_nodes = graph.N

        a, eig_sel, normalization, force_unsigned = _get_objective_matrices_and_eig_selector(graph, self.objective,
                                                                                               self.num_clusters,
                                                                                               self.force_unsigned)
        if labels is None:
            n_eigs = self.n_classes

            v0 = np.random.rand(min(a.shape))

            val, vec = eigsh(A=a.astype('float'), k=n_eigs, which=eig_sel, v0=v0)

            x = vec * normalization

            l_est = kmeans.predict(x, num_clusters=self.n_classes, **kwargs)

            l_est = l_est.astype(int)

        else:
            if eig_sel == 'LM':
                obj_matrix = a
            else:
                eig_upper_bound = max(abs(a).sum(1))
                obj_matrix = eig_upper_bound * eye(a.shape[0]) - a

            B, c = _labels_to_lin_const(labels, num_nodes=num_nodes, num_clusters=self.n_classes)

            if self.multiclass_method == 'joint':
                x = _joint_multiclass(obj_matrix, B=B, c=c, random_init=self.random_init, use_qr=False,
                                     return_intermediate=self.save_intermediate, eps=self.eps, t_max=self.t_max)
            if self.multiclass_method == 'qr':
                x = _joint_multiclass(obj_matrix, B=B, c=c, random_init=self.random_init, use_qr=True,
                                     return_intermediate=self.save_intermediate, eps=self.eps, t_max=self.t_max)
            elif self.multiclass_method == 'sequential':
                x = _sequential_multiclass(obj_matrix, B=B, c=c, random_init=self.random_init,
                                          return_intermediate=self.save_intermediate, eps=self.eps, t_max=self.t_max)

            if self.save_intermediate:
                self.intermediate_results = x.copy()
                x = x[0]

            l_est = seeded_kmeans.cluster(x, num_clusters=self.num_clusters, labels=labels)
            self.l_est = l_est.astype(int)

        returl self.l_est
