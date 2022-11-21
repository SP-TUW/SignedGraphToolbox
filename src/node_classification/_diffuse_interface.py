import numpy as np
import scipy.sparse as sps
from numpy import max
from numpy.linalg import norm
from scipy.sparse import diags

from src.node_classification._node_learner import NodeLearner
from src.tools.projections import simplex_projection


def _multiclass_GL_minimization(diffusion_parameter, stepsize, num_classes, label_weight, labels, eig_vals, eig_vecs,
                                eps, t_max, verbosity):
    '''
    Implementation of Multiclass GL algorithm (Fig. 1. on page 1605) of :cite:p:`Gar14Diffuse`

    :param diffusion_parameter: eps
    :param stepsize: dt
    :param num_classes: K
    :param label_weight: mu
    :param labels: used as a proxy for Uhat
    :param eig_val: [Nx1] ndarray Lambda
    :param eig_vec: x
    :param eps: threshold for stopping criterion
    :param t_max: maximum number of iteration
    :return:
    :param u: a local optimum of the optimization problem
    '''
    # N_D
    num_nodes = eig_vecs.shape[0]
    # N_e
    num_eig = eig_vecs.shape[1]

    c = label_weight + 3 / diffusion_parameter
    y = 1 / (1 + c * stepsize + diffusion_parameter * stepsize * eig_vals) * eig_vecs.T

    # u = np.random.rand(num_nodes, num_classes)
    u = np.ones((num_nodes, num_classes)) / num_classes
    u = simplex_projection(u, a=1, axis=1)
    u[labels['i'], :] = 0
    u[labels['i'], labels['k']] = 1

    E = np.eye(num_classes)
    ind_not_in_k = (1 - E).astype(bool)

    mu = np.zeros(num_nodes)
    mu[labels['i']] = label_weight
    mu = diags(mu, offsets=0, shape=(num_nodes, num_nodes))

    t = 0
    converged = False
    while not converged:
        u_old = u.copy()
        t += 1
        norm_u_minus_e = np.zeros((num_nodes, num_classes))
        T = np.zeros((num_nodes, num_classes))
        for k in range(num_classes):
            norm_u_minus_e[:, k] = norm(u - E[k, :], ord=1, axis=1)
        prod_u_all = np.prod(norm_u_minus_e, axis=1)
        prod_u = np.zeros((num_nodes, num_classes))
        for k in range(num_classes):
            prod_u[:, k] = prod_u_all * np.prod(norm_u_minus_e[:, ind_not_in_k[k, :]], axis=1)
        for k in range(num_classes):
            T[:, k] = 1 / 2 ** (2 * num_classes - 1) * (np.sum(prod_u[:, ind_not_in_k[k, :]], axis=1) - prod_u[:, k])

        u_label_diff = u.copy()
        u_label_diff[labels['i'], :] -= E[labels['k'], :]
        z = y.dot((1 + c * stepsize) * u - stepsize / (2 * diffusion_parameter) * T - stepsize * mu.dot(u_label_diff))
        u = eig_vecs.dot(z)
        u = simplex_projection(u, a=1, axis=1)
        norm_diff = norm(u - u_old, axis=1)
        norm_val = norm(u, axis=1)
        converged = max(norm_diff) / max(norm_val) <= eps
        if verbosity > 0:
            print('\rt:{t:d}, max_diff/max_abs={c}'.format(t=t, c=max(norm_diff) / max(norm_val)), end='')
        if not converged and t >= t_max:
            break
    if verbosity > 0:
        print()
    return u


class DiffuseInterface(NodeLearner):
    def __init__(self, num_classes=2, verbosity=1, save_intermediate=None, num_eig=20, objective='sym', eps=1e-6,
                 t_max=2 * 1e3, diffusion_parameter=1e-1, stepsize=1e-1, label_weight=1e3, use_full_matrix=False):
        self.num_eig = num_eig
        known_objectives = ['sym', 'am', 'lap', 'sponge']
        if objective not in known_objectives:
            raise ValueError('unknown objective \'{s}\'\nKnown objectives are {o}'.format(s=objective, o=known_objectives))
        self.which = objective
        self.eps = eps
        self.t_max = t_max
        self.diffusion_parameter = diffusion_parameter
        self.stepsize = stepsize
        self.label_weight = label_weight
        self.use_full_matrix = use_full_matrix
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        if self.num_eig < 0:
            use_full_matrix = True
            num_eig = graph.N
        else:
            num_eig = self.num_eig

        if self.which == 'sym':
            L = graph.get_signed_sym_laplacian()
            print('GL using the symmetric normalized signed Laplacian')
        elif self.which == 'am':
            L = graph.get_signed_am_laplacian()
            print('GL using the arithmetic mean Laplacian')
        elif self.which == 'lap':
            L = graph.get_signed_laplacian()
            print('GL using the signed Laplacian')
        elif self.which == 'sponge':
            matrix_numerator, matrix_denominator = graph.get_sponge_matrices()
            a_ = matrix_numerator
            b_ = matrix_denominator
            eig_sel = 'SM'
            force_unsigned = False

            w, v = np.linalg.eigh(matrix_denominator.A)
            denom_inv_sqrt = (v/np.sqrt(w)).dot(v.T)
            L = denom_inv_sqrt.dot(matrix_numerator.dot(denom_inv_sqrt))
            print('GL using the SPONGE matrix')
        else:
            raise ValueError('unknown objective ''{s}'''.format(s=self.which))

        if not self.use_full_matrix:
            eig_vals, eig_vecs = sps.linalg.eigsh(L, k=num_eig, which='SM')
        else:
            eig_vals, eig_vecs = np.linalg.eigh(L.A)
            i_sort = np.argsort(eig_vals)
            eig_vals = eig_vals[i_sort[:num_eig]]
            eig_vecs = eig_vecs[:, i_sort[:num_eig]]
        eig_vals = eig_vals[:, np.newaxis]

        u = _multiclass_GL_minimization(diffusion_parameter=self.diffusion_parameter, stepsize=self.stepsize,
                                        num_classes=self.num_classes, label_weight=self.label_weight,
                                        labels=labels, eig_vals=eig_vals, eig_vecs=eig_vecs, eps=self.eps,
                                        t_max=self.t_max, verbosity=self.verbosity)

        u[labels['i'], :] = 0
        u[labels['i'], labels['k']] = 1

        l_est = np.argmax(u, axis=1)

        self.embedding = u
        self.normalized_embedding = u

        return l_est
