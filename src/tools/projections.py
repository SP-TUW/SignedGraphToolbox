import numpy as np
import warnings




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