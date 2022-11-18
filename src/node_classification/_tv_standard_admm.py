import warnings

import numpy as np
import scipy.sparse as sps

from src.metric_learning import SeededKMeans
from src.tools.projections import min_norm_simplex_projection, label_projection
from ._node_learner import NodeLearner


def _standard_admm_x_update(x_in, Q, P, constants, labels, t_max, eps, backtracking_stepsize, backtracking_tau_0,
                            backtracking_param, normalize):
    N = x_in.shape[0]
    K = x_in.shape[1]

    x_in = x_in.copy()

    x_tp1 = min_norm_simplex_projection(x_in, min_norm=K - 2, sum_target=2 - K, min_val=-1)
    f_tp1 = np.sum((Q.dot(x_tp1) + P) * x_tp1)

    tau_0 = backtracking_tau_0

    converged = False
    t = 0
    while not converged:
        t += 1
        x_t = x_tp1.copy()
        f_t = f_tp1
        grad = 2 * Q.dot(x_t) + P
        grad = grad - np.mean(grad, axis=1, keepdims=True)
        slope = -np.linalg.norm(grad)
        backtracking_converged = False
        t_inner = 0
        tau = tau_0
        while not backtracking_converged:
            t_inner += 1

            x = x_t - tau * grad
            x_ = label_projection(x, labels=labels)
            x_tp1 = min_norm_simplex_projection(x_, min_norm=K - 2, sum_target=2 - K, min_val=-1)
            f_tp1 = np.sum((Q.dot(x_tp1) + P) * x_tp1)

            a = -backtracking_param * tau * slope
            b = f_t - f_tp1
            backtracking_converged = a <= b
            if (tau == 0.0 or np.linalg.norm(x_t - x_tp1) < eps ** 2) and not backtracking_converged:
                x_tp1 = x_t
                # warnings.warn('no improvement found in backtracking')
                break
            tau *= backtracking_stepsize
        dv = np.linalg.norm(x_t - x_tp1)
        dv_max = eps
        converged = dv < dv_max
        if t > t_max:
            print('x update did not converge')
            break

    x_out = x_tp1
    f_tp1 = f_tp1

    return x_out, {'f_p': f_tp1, 'f_d': -float('inf')}


def _standard_admm_y_update(beta, v, p):
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


def _run_standard_admm(graph, num_classes, p, beta, labels, x0, t_max, t_max_inner, t_max_no_change, eps, eps_admm,
                       eps_inner,
                       backtracking_stepsize, backtracking_tau_0, backtracking_param, laplacian_scaling,
                       pre_iteration_version, normalize_x,
                       penalty_strat_threshold, penalty_strat_scaling, penalty_strat_init_check,
                       penalty_strat_interval_factor, verbosity):
    gradient_matrix, divergence_matrix = graph.get_gradient_matrix(p=p, return_div=True)

    def grad(x_):
        g = gradient_matrix.dot(x_)
        return g

    def div(z_):
        d_ = divergence_matrix.dot(z_)
        return d_

    x_update_args = {'eps': eps_inner, 't_max': t_max_inner, 'backtracking_stepsize': backtracking_stepsize,
                     'backtracking_tau_0': backtracking_tau_0, 'backtracking_param': backtracking_param}

    gradient_matrix = graph.get_gradient_matrix(p=p, return_div=False)
    Q = gradient_matrix.T.dot(gradient_matrix) / 2

    eps_abs = eps_admm
    eps_rel = eps_admm

    x__ = label_projection(x0.copy(), labels=labels)
    x0_ = min_norm_simplex_projection(x__, min_norm=0, sum_target=2 - num_classes, min_val=-1)
    x = x0_.copy()

    if pre_iteration_version == 0:
        y = 0 * _standard_admm_y_update(beta, beta * grad(x), p)
        z = 0 * beta * (grad(x) - y)
    elif pre_iteration_version == 1:
        y = _standard_admm_y_update(beta, beta * grad(x), p)
        z = beta * (grad(x) - y)
    elif pre_iteration_version == 2:
        y = _standard_admm_y_update(beta, 2 * grad(x), p)
        z = 0 * beta * (grad(x) - y)

    l_est = np.argmax(x, axis=1)
    num_nodes = x0.shape[0]

    dx = []
    dy = []
    dz = []
    fx_pd = {'p': [], 'd': []}
    converged = False
    t = 0
    t_since_last = 0
    t_check_rho = penalty_strat_init_check
    while not converged:
        x_old = x.copy()
        y_old = y.copy()
        z_old = z.copy()
        l_est_old = l_est.copy()

        t += 1

        d = div(z - beta * y) / (2 * beta)  # calc_d(y_old, z_old, beta, div)
        P = -2 * d
        x, fx_pd_ = _standard_admm_x_update(x_old, Q=Q, P=P, labels=labels, normalize=normalize_x, **x_update_args)

        fx_pd['p'].append(fx_pd_['f_p'])
        fx_pd['d'].append(fx_pd_['f_d'])
        l_est = np.argmax(x, axis=1)

        v = z_old + beta * grad(x)
        y = _standard_admm_y_update(beta, v, p)

        z = z_old + beta * (grad(x) - y)

        r = grad(x) - y
        s = beta * div(y - y_old)

        norm_r = np.linalg.norm(r)
        norm_s = np.linalg.norm(s)

        ePri = np.sqrt(num_classes) * num_nodes * eps_abs + eps_rel * np.linalg.norm(grad(x))
        eDual = np.sqrt(num_classes * num_nodes) * eps_abs + eps_rel * np.linalg.norm(div(y))
        if t >= t_check_rho:
            if norm_r * eDual >= norm_s * ePri * penalty_strat_threshold:
                beta = beta * penalty_strat_scaling

                print('beta = {b}'.format(b=beta))
            elif norm_r * eDual <= norm_s * ePri / penalty_strat_threshold:
                beta = beta / penalty_strat_scaling
                print('beta = {b}'.format(b=beta))
            t_check_rho = t_check_rho * penalty_strat_interval_factor
        cont = (norm_r > ePri or norm_s > eDual)

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
            print(
                '\rt={t}, dx_={dx:.3e}, r={r:.3e}, s={s:.3e}, {n} changes for {t_since} iterations'.format(t=t, dx=dx_,
                                                                                                           r=norm_r,
                                                                                                           s=norm_s,
                                                                                                           n=num_changes,
                                                                                                           t_since=t_since_last),
                end='')
        converged = not cont and (
                dx_ <= eps * np.sqrt(x.size) and dy_ <= eps * np.sqrt(y.size) and dz_ <= eps * np.sqrt(
            z.size)) or t_since_last >= t_max_no_change
        # converged = dx_ <= eps
        if t >= t_max and not converged:
            warnings.warn('TVNC did not converge')
            break
    if verbosity > 0:
        print('\r')
    return x, converged, dx, dy, dz, fx_pd


class TvStandardADMM(NodeLearner):
    def __init__(self, num_classes=2, verbosity=0, save_intermediate=None,
                 penalty_parameter=1000, p=1,
                 t_max=2000, t_max_inner=10000, t_max_no_change=None, eps=1e-3, eps_admm=1e-4, eps_inner=1e-5,
                 backtracking_stepsize=1 / 2, backtracking_tau_0=0.01, backtracking_param=1 / 2,
                 penalty_strat_threshold=float('inf'), penalty_strat_scaling=1, penalty_strat_init_check=32,
                 penalty_strat_interval_factor=2,
                 laplacian_scaling=1, pre_iteration_version=0, normalize_x=False):
        self.t_max = t_max
        self.penalty_parameter = penalty_parameter
        self.beta = penalty_parameter
        self.p = p
        self.t_max_inner = t_max_inner
        self.t_max_no_change = t_max_no_change
        self.eps = eps
        self.eps_admm = eps_admm
        self.eps_inner = eps_inner
        self.backtracking_stepsize = backtracking_stepsize
        self.backtracking_tau_0 = backtracking_tau_0
        self.backtracking_param = backtracking_param
        self.laplacian_scaling = laplacian_scaling
        self.pre_iteration_version = pre_iteration_version
        self.normalize_x = normalize_x
        self.penalty_strat_threshold = penalty_strat_threshold
        self.penalty_strat_scaling = penalty_strat_scaling
        self.penalty_strat_init_check = penalty_strat_init_check
        self.penalty_strat_interval_factor = penalty_strat_interval_factor
        super().__init__(num_classes=num_classes, verbosity=verbosity, save_intermediate=save_intermediate)

    def estimate_labels(self, graph, labels=None, guess=None):
        num_nodes = graph.num_nodes
        if guess is None:
            x0 = np.ones((num_nodes, self.num_classes)) + np.random.standard_normal(
                (num_nodes, self.num_classes)) / 100
        else:
            x0 = -np.ones((num_nodes, self.num_classes))
            x0[range(num_nodes), guess] = 1

        if labels is not None:
            x0[labels['i'], :] = -1
            x0[labels['i'], labels['k']] = 1

        if self.t_max_no_change is None:
            t_max_no_change = self.t_max
        elif self.t_max_no_change == 'auto':
            t_max_no_change = np.sqrt(num_nodes * self.num_classes)
        else:
            t_max_no_change = self.t_max_no_change

        x, converged, dx, dy, dz, fx_pd = _run_standard_admm(graph=graph, num_classes=self.num_classes, x0=x0,
                                                             labels=labels,
                                                             verbosity=self.verbosity, t_max=self.t_max,
                                                             t_max_inner=self.t_max_inner,
                                                             t_max_no_change=t_max_no_change,
                                                             beta=self.beta, eps=self.eps, eps_admm=self.eps_admm,
                                                             eps_inner=self.eps_inner, p=self.p,
                                                             backtracking_param=self.backtracking_param,
                                                             backtracking_tau_0=self.backtracking_tau_0,
                                                             backtracking_stepsize=self.backtracking_stepsize,
                                                             penalty_strat_threshold=self.penalty_strat_threshold,
                                                             penalty_strat_scaling=self.penalty_strat_scaling,
                                                             penalty_strat_init_check=self.penalty_strat_init_check,
                                                             penalty_strat_interval_factor=self.penalty_strat_interval_factor,
                                                             laplacian_scaling=self.laplacian_scaling,
                                                             pre_iteration_version=self.pre_iteration_version,
                                                             normalize_x=self.normalize_x)

        kMeans = SeededKMeans(num_classes=self.num_classes, verbose=self.verbosity)
        l_est = kMeans.estimate_labels(x, labels=labels)
        exit_code = 0 if converged else -1
        intermediate_results = {'exit_code': exit_code,
                                'dx': dx,
                                'dy': dy,
                                'dz': dz,
                                'fx_pd': fx_pd}

        self.embedding = x
        self.normalized_embedding = (x + 1) / 2
        self.intermediate_results = intermediate_results

        return l_est
