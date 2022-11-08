import numpy as np
import scipy.sparse
from numpy import sqrt
from scipy.linalg import qr
from scipy.sparse import csc_matrix, eye, issparse, diags
from scipy.sparse.linalg import spsolve, eigsh, lobpcg
from sklearn.cluster import KMeans

from src.metric_learning import SeededKMeans
from src.tools.projections import unitarization
from ._node_learner import NodeLearner


def _get_objective_matrices_and_eig_selector(graph, objective, num_classes):
    weight_matrix = graph.weights

    a_ = eye(graph.num_nodes)
    b_ = eye(graph.num_nodes)

    force_unsigned = True
    if objective == "RC":
        d = abs(weight_matrix).sum(axis=1).T
        d_nz = np.where(d == 0, 1, d)
        deg_matrix = diags(d_nz)
        a = deg_matrix - weight_matrix
        normalization = diags(np.ones(graph.num_nodes))
        eig_sel = 'SM'
    elif objective == "NC":
        d = abs(weight_matrix).sum(axis=1).T
        d_nz = np.where(d == 0, 1, d)
        deg_matrix = diags(d_nz)
        a = deg_matrix - weight_matrix
        dd = sqrt(d)
        dd = np.where(dd == 0, 1, dd).squeeze()
        inv_sqrt_deg_matrix = diags(1 / dd)
        a = inv_sqrt_deg_matrix.dot(a.dot(inv_sqrt_deg_matrix))

        normalization = diags(np.array(1 / dd))
        eig_sel = 'SM'
    elif objective == "BNC" or objective == "BNC_INDEF":
        inv_sqrt_sig_deg = (1 / sqrt(abs(weight_matrix).sum(axis=1)).T).A.squeeze()
        inv_sqrt_sig_deg_matrix = diags(inv_sqrt_sig_deg)
        neg_weight_matrix = -weight_matrix.minimum(0)
        neg_deg = neg_weight_matrix.sum(axis=1).A.squeeze()
        neg_deg_matrix = diags(neg_deg)
        a = inv_sqrt_sig_deg_matrix.dot((neg_deg_matrix + weight_matrix)).dot(inv_sqrt_sig_deg_matrix)
        if objective != "BNC_INDEF":
            a = a+eye(graph.num_nodes)
        normalization = diags(np.ones(graph.num_nodes))
        eig_sel = 'LM'
        force_unsigned = False
    elif objective == "SPONGE":
        matrix_numerator, matrix_denominator = graph.get_sponge_matrices()
        a_ = matrix_numerator
        b_ = matrix_denominator
        eig_sel = 'SM'
        force_unsigned = False

        w, v = np.linalg.eigh(matrix_denominator.A)
        denom_inv_sqrt = (v/np.sqrt(w)).dot(v.T)
        a = denom_inv_sqrt.dot(matrix_numerator.dot(denom_inv_sqrt))
        normalization = denom_inv_sqrt

    return a, eig_sel, normalization, force_unsigned, a_, b_


def _labels_to_lin_const(labels, num_nodes, num_classes):
    num_labels = len(labels['i'])

    labels_i = np.array(labels['i'])
    labels_k = np.array(labels['k'])
    B = csc_matrix((np.ones(num_labels), (np.arange(num_labels), labels_i)), shape=(num_labels, num_nodes))
    c = np.zeros((num_labels, num_classes))
    c[np.arange(num_labels), labels_k] = 1
    c = unitarization(c) * np.sqrt(num_labels / num_nodes)

    return B, c


def _joint_multiclass(obj_matrix, B, c, random_init, return_intermediate, use_qr, eps, t_max, verbosity):
    '''
    This function implements our joint multiclass algorithm where the linear constraints are given by the labels

    In the case of c.shape[1]==1 it is equivalent to the method from :cite:p:`Xu09LinConstCut`

    :param obj_matrix: matrix for quadratic objective. Needs to be positive semi-definite
    :param B, c: the linear constraints on data of the form BX=c
    :return: vector v which maximizes the optimization problem.
    '''

    num_labels = c.shape[0]
    if len(c.shape) > 1:
        num_classes = c.shape[1]
    else:
        num_classes = 1
    num_nodes = obj_matrix.shape[0]

    if use_qr:
        def normalization(X_):
            X, R = qr(X_,mode='economic')
            S = np.diag(np.sign(np.diag(R)))
            X = X.dot(S)
            R = S.dot(R)
            return X
    else:
        def normalization(X_):
            X = unitarization(X_)
            return X

    obj_list = []

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
        PAv_unit = normalization(PAv)
        u = gamma * PAv_unit
        v = u + n0
        obj_list.append(np.trace(v.T.dot(obj_matrix.dot(v))))
        t += 1
        diff = np.linalg.norm(v - v_old)
        converged = diff < eps * sqrt((num_nodes - num_labels) * num_classes)
        print('\r{t}'.format(t=t),end='')
        if not converged and t >= t_max:
            break

    if verbosity>0:
        print('finished with obj={o} after t={t} with diff={d}'.format(o=obj_list[-1],t=t,d=diff))

    if return_intermediate:
        return v, obj_list, obj_matrix.copy()
    else:
        return v


def _sequential_multiclass(obj_matrix, B, c, random_init, return_intermediate, eps, t_max, verbosity):
    num_classes = c.shape[1]
    num_nodes = obj_matrix.shape[0]

    obj_list = []

    v = np.zeros((num_nodes, num_classes))
    for k in range(num_classes):
        if k > 0:
            B_ = np.vstack([B.A, v[:, :k].T])
            c_ = np.concatenate((c[:, k], np.zeros((k))), axis=0)[:, None]
        else:
            B_ = B.A.copy()
            c_ = c[:, k][:, None]

        r_val = _joint_multiclass(obj_matrix, B_, c_, random_init=random_init, return_intermediate=return_intermediate,
                                 eps=eps, t_max=t_max, use_qr=False, verbosity=verbosity)

        if return_intermediate:
            v[:, k] = r_val[0]
            obj_list.append(r_val[1])
        else:
            v[:, k] = r_val[:, 0]

    if return_intermediate:
        return v, obj_list, obj_matrix.copy()
    else:
        return v


class SpectralLearning(NodeLearner):

    def __init__(self, num_classes=2, verbosity=0, save_intermediate=False, objective='BNC', multiclass_method='joint', random_init=False, eps=1e-5, t_max=int(1e5)):
        self.num_classes = num_classes
        self.objective = objective
        self.multiclass_method=multiclass_method
        self.random_init=random_init
        self.verbosity = verbosity
        self.eps=eps
        self.t_max = t_max
        self.save_intermediate = save_intermediate
        self.intermediate_results = None
        self.kmeans = None
        self.embedding = None
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        num_nodes = graph.num_nodes

        if self.num_classes is None:
            self.num_classes = graph.num_classes

        a, eig_sel, normalization, force_unsigned, a_, b_ = _get_objective_matrices_and_eig_selector(graph, self.objective,
                                                                                               self.num_classes)
        if labels is None or len(labels['i'])==0:
            n_eigs = self.num_classes

            v0 = np.random.randn(graph.num_nodes,n_eigs)

            val, x = eigsh(A=a.astype('float'), k=n_eigs, which=eig_sel, v0=v0[:,0])

        else:
            if eig_sel == 'LM':
                obj_matrix = a
            else:
                eig_upper_bound = max(abs(a).sum(1))
                if issparse(a):
                    obj_matrix = eig_upper_bound * eye(a.shape[0]) - a
                else:
                    obj_matrix = eig_upper_bound * np.eye(a.shape[0]) - a

            B, c = _labels_to_lin_const(labels, num_nodes=num_nodes, num_classes=self.num_classes)

            if self.multiclass_method == 'joint':
                x = _joint_multiclass(obj_matrix, B=B, c=c, random_init=self.random_init, use_qr=False,
                                     return_intermediate=self.save_intermediate, eps=self.eps, t_max=self.t_max, verbosity=self.verbosity)
            if self.multiclass_method == 'qr':
                x = _joint_multiclass(obj_matrix, B=B, c=c, random_init=self.random_init, use_qr=True,
                                     return_intermediate=self.save_intermediate, eps=self.eps, t_max=self.t_max, verbosity=self.verbosity)
            elif self.multiclass_method == 'sequential':
                x = _sequential_multiclass(obj_matrix, B=B, c=c, random_init=self.random_init,
                                          return_intermediate=self.save_intermediate, eps=self.eps, t_max=self.t_max, verbosity=self.verbosity)

            if self.save_intermediate:
                self.intermediate_results = x
                x = x[0]

        x = normalization.dot(x)
        self.kmeans = SeededKMeans(num_classes=self.num_classes)
        l_est = self.kmeans.estimate_labels(data=x, labels=labels)

        self.embedding = x
        self.normalized_embedding = x + 1/2
        self.l_est = l_est.astype(int)

        return self.l_est
