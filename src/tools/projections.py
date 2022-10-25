import numpy as np
import warnings


def label_projection(x_in, labels, values=[-1, 1]):
    x_out = x_in.copy()
    if labels is not None:
        x_out[labels['i'], :] = values[0]
        x_out[labels['i'], labels['k']] = values[1]
    return x_out


def min_norm_simplex_projection(y, min_norm, sum_target, min_val, axis=-1, return_lags=False):
    assert len(y.shape) == 2 and (axis == -1 or axis == 1), 'currently only implemented for rows of matrices'
    N = y.shape[0]
    K = y.shape[axis]

    # parameters for affine transformation
    k = sum_target - K * min_val
    d = min_val

    # transform input
    z = (y - d) / k
    alpha = (min_norm - 2 * k * d - d ** 2 * K) / k ** 2

    r0 = np.floor(1 / alpha).astype(int)

    z_sort = -np.sort(-z, axis=axis)

    xi = np.ones((N,K))
    cond = np.ones((N,K), dtype=bool)
    r = np.arange(1,K+1)[None,:]
    z_cumsum = np.cumsum(z_sort,axis=axis)
    z_cumnorm = np.cumsum(z_sort ** 2, axis=axis)
    mean_zr = z_cumsum/r
    xi[:, r0:] = np.minimum(1, np.sqrt((z_cumnorm[:,r0:]-z_cumsum[:,r0:]**2/r[:,r0:])/(alpha - 1 / r[:,r0:])))
    nu = xi / r - mean_zr
    cond[:, :-1] = np.bitwise_and(z_sort[:, :-1] >= -nu[:, :-1], -nu[:, :-1] >= z_sort[:, 1:])
    r__ = np.argmax(cond, axis=axis)
    xi_r = np.take_along_axis(xi, r__[:, None], axis=axis)
    nu_r = np.take_along_axis(nu, r__[:, None], axis=axis)
    is_randomized = np.sum(cond[:, :-1], axis=1) > 1
    x_ = np.maximum(0,(z+nu_r)/xi_r)

    if np.any(is_randomized):
        warnings.warn('at least one element needs randomization')
        randomized_indices = np.nonzero(is_randomized)

        if axis == -1:
            full_indices = randomized_indices[:axis] + (slice(None),)
        else:
            full_indices = randomized_indices[:axis] + (slice(None),) + randomized_indices[axis + 1:]
        z_randomized = z[full_indices]
        z_randomized += np.random.standard_normal(z_randomized.shape) / 100
        x_randomized = min_norm_simplex_projection(z_randomized, min_norm=alpha, sum_target=1, min_val=0, axis=axis)
        x_[full_indices] = x_randomized

    # transform output
    x = k * np.maximum(x_, 0) + d

    if return_lags:
        return x, xi, nu
    else:
        return x


def simplex_projection(x, a=1, axis=1):
    '''
    Implementation of the algorithm for projection onto a simplex from :cite:p:`Duc08Simplex`

    :param x: ndarray input which should be projected.
    :param a: scalar value for the sum constraint
    :param axis: scalar axis over which the projection should operate.
    :return: projection of x onto the simplex with the constraint numpy.sum(x,axis=axis)==a
    '''
    if a > 0:
        K = x.shape[axis]
        u = -np.sort(-x, axis=axis)  # sort descending
        cs = np.cumsum(u, axis=axis)
        shape_without_axis = np.array(x.shape)
        shape_without_axis[axis] = 1
        j = np.arange(K) + 1
        j = np.broadcast_to(j, x.shape)
        j = np.moveaxis(j, -1, axis)
        u_ = u - 1 / j * (cs - a)
        r = np.count_nonzero(u_ > 0, axis=axis, keepdims=True)
        # cumsum up to the r-th element
        cs_r = np.take_along_axis(cs, r - 1, axis=axis)
        l = (a - cs_r) / r
        return np.maximum(x + l, 0)
    else:
        raise ValueError('Projection is only possible for positive sum constraints')


def unitarization(x):
    '''
    Unitarization of the columns of x following section 4 of :cite:p:`Hig88MatrixProjections`

    :param x: N-by-K matrix/ndarray to be projected
    :return: projection of x such that x.T.dot(x) is the identity matrix
    '''
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    if s[-1] < 1e-6:
        warnings.warn("matrix is possibly not full rank")
    s_u = np.eye(u.shape[1], vh.shape[0])
    return u.dot(s_u.dot(vh))
