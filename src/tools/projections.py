import numpy as np
import warnings


def label_projection(x_in, labels, values=[-1,1]):
    x_out = x_in.copy()
    if labels is not None:
        x_out[labels['i'], :] = values[0]
        x_out[labels['i'], labels['k']] = values[1]
    return x_out


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
