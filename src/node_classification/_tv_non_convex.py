from ._node_learner import NodeLearner
from src.tools.projections import min_norm_simplex_projection, simplex_projection, label_projection
from src.metric_learning import SeededKMeans

import numpy as np
import scipy.sparse as sps


def get_lap_2overP(weights, p):
    weights_abs = abs(weights)
    signs = weights.sign()
    weights_2p = weights_abs.power(2 / p)
    in_deg_lp = np.squeeze(np.asarray(weights_2p.sum(axis=1)))
    in_deg_lp_lap = sps.diags(in_deg_lp) - signs.multiply(weights_2p)
    out_deg_lp = np.squeeze(np.asarray(weights_2p.sum(axis=0)))
    out_deg_lp_lap = sps.diags(out_deg_lp) - signs.multiply(weights_2p).T
    lap_2overP = 1 / 2 * (in_deg_lp_lap + out_deg_lp_lap)
    return lap_2overP


def get_hard_constants(weights, beta, p, labels, num_classes):
    lap_2overP = get_lap_2overP(weights, p)
    label_indicator = np.zeros(weights.shape[0],dtype=bool)
    if labels is not None:
        label_indicator[labels['i']] = True
    A = 4 * lap_2overP[np.bitwise_not(label_indicator), :]
    A = A[:, np.bitwise_not(label_indicator)]
    B_offset = 2*lap_2overP[np.bitwise_not(label_indicator),:].sum(1).dot(np.ones((1,num_classes)))
    if labels is not None:
        v_L = np.zeros((len(labels['i']),num_classes))
        v_L[:,labels['k']] = 1
        B_offset -= 4*lap_2overP[np.bitwise_not(label_indicator),:][:, label_indicator].dot(v_L)
    Q_chol = np.linalg.cholesky(lap_2overP.A).T
    constants = {'Q': lap_2overP,
                 'A': A,
                 'B_offset': B_offset.A,
                 'Q_chol': Q_chol,
                 'beta': beta}
    return constants


def calc_d_hard(y, z, beta, div):
    return div(z - beta * y)/beta


def x_objective_hard(x, d, constants):
    P = -d
    obj = 0
    for k in range(x.shape[1]):
        xk = x[:, k]
        obj += (constants['Q'].T.dot(xk) + P[:, k]).dot(xk)
    return obj


def x_grad_hard(x, d, constants):
    P = -d
    grad = 2 * constants['Q'].dot(x) + P
    return grad

def x_update_hard(x_in, d, constants, labels, t_max, eps, backtracking_stepsize, backtracking_tau_0,
                  backtracking_param=1):
    x = x_in.copy()

    is_unlabeled = np.ones(x_in.shape[0], dtype=bool)
    if labels is not None:
        is_unlabeled[labels['i']] = False
    v_in = (x_in[is_unlabeled, :] + 1) / 2

    A = constants['A']
    B = 2 * d[is_unlabeled, :] + constants['B_offset']
    v_tp1 = v_in.copy()

    v_tp1 = simplex_projection(v_tp1)

    f_tp1 = np.sum((A.dot(v_tp1) / 2 - B) * v_tp1)

    tau_0 = backtracking_tau_0

    converged = False
    t = 0
    tau = tau_0
    while not converged:
        v_t = v_tp1.copy()
        f_t = f_tp1
        t += 1
        grad = A.dot(v_t) - B
        slope = -np.linalg.norm(grad)
        backtracking_converged = False
        t_inner = 0
        tau = tau_0
        while not backtracking_converged:
            t_inner += 1
            v_ = v_t - tau * grad
            v_tp1 = min_norm_simplex_projection(v_, min_norm=1 / 2, sum_target=1, min_val=0)
            f_tp1 = np.sum((A.dot(v_tp1) / 2 - B) * v_tp1)
            a = -backtracking_param * tau * slope
            b = f_t - f_tp1
            backtracking_converged = a < b
            if (t_inner > 80 or tau == 0.0) and not backtracking_converged:
                v_tp1 = v_t
                break
            tau *= backtracking_stepsize
        dv = np.linalg.norm(v_t - v_tp1)
        dv_max = np.minimum(1, np.linalg.norm(v_tp1 - v_in)) * eps
        converged = dv <= dv_max or t > t_max
    x[is_unlabeled, :] = 2 * v_tp1 - 1
    return x, {'f_p': f_tp1, 'f_d': -float('inf')}, None


def y_update(beta, v, p):
    # minimize |y|_+^p- vy+ y^2*beta/2
    y = np.zeros(v.shape)
    v_neg = v < 0
    y[v_neg] = v[v_neg] / beta
    if p == 1:
        v_big = v > 1
        # everything in between results in 0
        y[v_big] = (v[v_big] - 1) / beta
    elif p == 2:
        v_pos = v > 0
        y[v_pos] = v[v_pos] / (2 + beta)
    elif p == 3:
        v_pos = v > 0
        y[v_pos] = (np.sqrt(beta ** 2 + 12 * v[v_pos]) - beta) / 6
    elif p > 1:
        raise ValueError('y_update for arbitrary p>0 not yet implemented')
    else:
        raise ValueError('p<1 not supported')
    return y



def objective_hard(x, p, gradient_matrix, labels):
    y = gradient_matrix.dot(x)
    z = np.zeros(y.shape)
    return lagrangian_hard(x, y, z, p, 0, gradient_matrix, labels)


def lagrangian_hard(x, y, z, p, beta, gradient_matrix, labels):
    num_classes = x.shape[1]
    pos_grad = np.maximum(y, 0)
    tv = np.linalg.norm(pos_grad, ord=p) ** p
    # use np.allclose here to avoid problems with numeric imprecision in sum_projection
    is_in_simplex = np.all(x <= 1) and np.all(x >= -1) and np.allclose(np.sum(x, axis=1), 2 - num_classes)
    # use equality here because label constraints are always enforced exactly
    if labels is not None:
        labels_correct = np.all(x[labels['i'], labels['k']] == 1)
    else:
        labels_correct = True
    min_norm_sufficient = np.all(np.linalg.norm(x, axis=1) >= num_classes - 2)
    char_func = 0 if is_in_simplex and labels_correct and min_norm_sufficient else float('inf')
    grad_x = gradient_matrix.dot(x)
    inner_prod = np.sum(z * (grad_x - y))
    norm_term = beta / 2 * np.linalg.norm(grad_x - y) ** 2
    return tv + char_func + inner_prod + norm_term


def nc_admm(graph, num_classes, p, a, b, c, beta, labels,
            x0, t_max, t_max_inner, t_max_no_change,
            eps, eps_inner, backtracking_stepsize, backtracking_tau_0, backtracking_param,
            verbosity):
    gradient_matrix, divergence_matrix = graph.get_gradient_matrix(p=p, return_div=True)

    def grad(x):
        g = gradient_matrix.dot(x)
        return g

    def div(z):
        d = divergence_matrix.dot(z)
        return d

    x_update_args = {'eps': eps_inner, 't_max': t_max_inner,
                     'backtracking_stepsize': backtracking_stepsize, 'backtracking_tau_0': backtracking_tau_0,
                     'backtracking_param': backtracking_param,
                     'constants': get_hard_constants(graph.weights, beta=beta, p=p, labels=labels,
                                                     num_classes=num_classes)}

    def x_update(x_, y_, z_, dual_init):
        d = calc_d_hard(y_, z_, beta, div)
        return x_update_hard(x_, d, labels=labels, **x_update_args)

    def x_projection(x_):
        x__ = label_projection(x_, labels=labels)
        return min_norm_simplex_projection(x__, min_norm=num_classes-2, sum_target=2-num_classes, min_val=-1)

    def objective(x_):
        return objective_hard(x_, p, gradient_matrix, labels)

    def lagrangian(x_, y_, z_):
        return lagrangian_hard(x_, y_, z_, p, beta, gradient_matrix, labels)

    eps_abs = 1e-4
    eps_rel = 1e-4

    mu = 10
    eta = 2

    t_check_rho = 32
    i_check = 2

    x0_ = x_projection(x0.copy() / beta)
    # x0_ = x_projection(x0.copy())
    x = x0_.copy()
    l_est = np.argmax(x, axis=1)
    x_old = np.zeros(x.shape)
    fx = objective(x)
    num_nodes = x0.shape[0]
    num_edges = gradient_matrix.shape[0]
    # z = np.sqrt(eps) * np.random.randn(num_edges, num_classes)
    y = 0 * grad(x)  # np.zeros(z.shape)#y_update(beta, grad(x), p)
    # y = y_update(beta, 0*grad(x), p)  # np.zeros(z.shape)#y_update(beta, grad(x), p)
    z = np.zeros(y.shape)
    lag = lagrangian(x, y, z)

    dx = []
    dy = []
    dz = []
    fx_pd = {'p': [], 'd': []}
    converged = False
    t = 0
    t_since_last = 0
    dual_init = None
    while not converged:
        x_old = x.copy()
        y_old = y.copy()
        z_old = z.copy()
        l_est_old = l_est.copy()

        fx_old = fx
        lag_old = lag

        t += 1

        x, fx_pd_, dual_init = x_update(x_old, y_old, z_old, dual_init)
        fx_pd['p'].append(fx_pd_['f_p'])
        fx_pd['d'].append(fx_pd_['f_d'])
        l_est = np.argmax(x, axis=1)

        v = z_old + beta * grad(x)
        y = y_update(beta, v, p)

        z = z_old + beta * (grad(x) - y)

        r = grad(x) - y
        s = beta * div(y - y_old)

        norm_r = np.linalg.norm(r)
        norm_s = np.linalg.norm(s)

        # rtp1 = norm(Rtp1) / rho
        # stp1 = norm(Stp1)
        ePri = np.sqrt(num_classes) * num_nodes * eps_abs + eps_rel * np.linalg.norm(grad(x))
        eDual = np.sqrt(num_classes * num_nodes) * eps_abs + eps_rel * np.linalg.norm(div(y))
        if t >= t_check_rho:
            if norm_r * eDual >= norm_s * ePri * mu:
                beta = beta * eta
            elif norm_r * eDual <= norm_s * ePri / mu:
                beta = beta / eta
            t_check_rho = t_check_rho * i_check
        cont = (norm_r > ePri or norm_s > eDual)

        fx = objective(x)
        lag = lagrangian(x, y, z)

        dx_ = np.linalg.norm(x - x_old)
        dy_ = np.linalg.norm(y - y_old)
        dz_ = np.linalg.norm(z - z_old)
        dx.append(dx_)
        dz.append(dy_)
        dx.append(dz_)
        if verbosity > 0:
            num_changes = np.sum(l_est_old != l_est)
            if num_changes != 0:
                t_since_last = 0
            else:
                t_since_last += 1
            print('\rt={t}, dx_={dx}, {n} changes for {t_since} iterations'.format(t=t, dx=dx_, n=num_changes,
                                                                                   t_since=t_since_last), end='')
        converged = not cont and (
                dx_ <= eps * np.sqrt(x.size) and dy_ <= eps * np.sqrt(y.size) and dz_ <= eps * np.sqrt(
            z.size))  # or t_since_last >= t_max_no_change
        if t >= t_max and not converged:
            break
    if verbosity > 0:
        print('\r')
    return x, converged, dx, dy, dz, fx_pd


class TvNonConvex(NodeLearner):
    def __init__(self, num_classes=2, verbosity=0, save_intermediate=None,
                 penalty_parameter=100, p=1, a=1, b=100, c=20000,
                 t_max=10000, t_max_inner=10000, t_max_no_change=None, eps=1e-3, eps_inner=1e-5,
                 backtracking_stepsize=1 / 2, backtracking_tau_0=1 / 2, backtracking_param=1 / 2):
        self.t_max = t_max
        self.penalty_parameter = penalty_parameter
        self.beta = penalty_parameter
        self.p = p
        self.a = a
        self.b = b
        self.c = c
        self.t_max_inner = t_max_inner
        self.t_max_no_change = t_max_no_change
        self.eps = eps
        self.eps_inner = eps_inner
        self.backtracking_stepsize = backtracking_stepsize
        self.backtracking_tau_0 = backtracking_tau_0
        self.backtracking_param = backtracking_param
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        num_nodes = graph.num_nodes
        if guess is None:
            x0 = np.zeros((num_nodes, self.num_classes)) + np.random.standard_normal(
                (num_nodes, self.num_classes)) / 100
        else:
            x0 = -np.ones((num_nodes, self.num_classes))
            x0[range(num_nodes), guess] = 1

        if labels is not None:
            x0[labels['i'], :] = -1
            x0[labels['i'], labels['k']] = 1

        if self.t_max_no_change is None:
            t_max_no_change = np.sqrt(num_nodes * self.num_classes)
        else:
            t_max_no_change = self.t_max_no_change

        x, converged, dx, dy, dz, fx_pd = nc_admm(graph=graph, num_classes=self.num_classes, x0=x0, a=self.a,
                                                  b=self.b, c=self.c, labels=labels, verbosity=self.verbosity,
                                                  t_max=self.t_max, t_max_inner=self.t_max_inner,
                                                  t_max_no_change=t_max_no_change, beta=self.beta, eps=self.eps,
                                                  eps_inner=self.eps_inner, p=self.p,
                                                  backtracking_param=self.backtracking_param,
                                                  backtracking_tau_0=self.backtracking_tau_0,
                                                  backtracking_stepsize=self.backtracking_stepsize)

        # l_est = np.argmax(x, axis=1)
        kMeans = SeededKMeans(num_classes=self.num_classes, verbose=self.verbosity)
        l_est = kMeans.estimate_labels(x, labels=labels)
        exit_code = 0 if converged else -1
        intermediate_results = {'exit_code': exit_code,
                                'dx': dx,
                                'dy': dy,
                                'dz': dz,
                                'fx_pd': fx_pd}

        self.embedding = x
        self.normalized_embedding = (x+1)/2
        self.intermediate_results = intermediate_results

        return l_est
