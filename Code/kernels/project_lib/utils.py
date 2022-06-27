import numpy as np
from scipy import stats

def svds(X,k):
    U, s, V = np.linalg.svd(X,full_matrices=False)
    return U[:,:k], s[:k] , V[:k,:]

def construct_cov(d, v):
    """Reconstruct covariance matrix from eigenvalues and inverted eigenvectors of covariance matrix

    :param d: Eigenvalues of covariance matrix
    :param v: Eigenvectors of covariance matrix
    :return: Reconstructed covariance matrix
    """
    v_tpose = np.linalg.inv(v)
    d_mat = np.diag(d)
    return v.dot(d_mat.dot(v_tpose))


def get_sqrt_cov(cov):
    d, v = np.linalg.eigh(cov)
    return construct_cov(np.sqrt(d), v)


def winsorization(a):
    N = a.shape[1]
    a_mean = stats.trim_mean(a, 0.1, axis=1)
    a_mad = stats.median_abs_deviation(a,axis=1)
    a_lo = a_mean - 5 * a_mad
    a_hi = a_mean + 5 * a_mad
    a_hi = np.array([a_hi ]* N).T
    a_lo = np.array([a_lo ]* N).T
    cond1 = a > a_hi
    cond2 = a < a_lo
    out = np.where(cond2, a_lo, np.where(cond1, a_hi, a))
    return out


def cov_calc(spectrum, V):
    cov = V.dot(np.diag(spectrum)).dot(V.T)
    return 0.5 * (cov + cov.T)


def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr, std


def corr2cov(corr, std):
    cov = corr
    cov *= std
    cov *= std.reshape(-1, 1)
    return cov


def __valid(a):
    """
    Construct the valid subset of a (correlation) matrix a
    :param a: n x n matrix

    :return: Tuple of a boolean vector indicating if row/column is valid and the valid subset of the matrix
    """
    # make sure a  is quadratic
    assert a.shape[0] == a.shape[1]
    v = np.isfinite(np.diag(a))
    return v, a[:, v][v]


# that's somewhat not needed...
def a_norm(vector, a=None):
    """
    Compute the a-norm of a vector
    :param vector: the n x 1 vector
    :param a: n x n matrix
    :return:
    """
    if a is None:
        return np.linalg.norm(vector[np.isfinite(vector)], 2)

    # make sure a is quadratic
    assert a.shape[0] == a.shape[1]
    # make sure the vector has the right number of entries
    assert vector.size == a.shape[0]

    v, mat = __valid(a)

    if v.any():
        return np.sqrt(np.dot(vector[v], np.dot(mat, vector[v])))
    else:
        return np.nan


def inv_a_norm(vector, a=None):
    """
    Compute the a-norm of a vector
    :param vector: the n x 1 vector
    :param a: n x n matrix
    :return:
    """
    if a is None:
        return np.linalg.norm(vector[np.isfinite(vector)], 2)

    # make sure a is quadratic
    assert a.shape[0] == a.shape[1]
    # make sure the vector has the right number of entries
    assert vector.size == a.shape[0]

    v, mat = __valid(a)

    if v.any():
        return np.sqrt(np.dot(vector[v], np.linalg.solve(mat, vector[v])))
    else:
        return np.nan


def solve(a, b):
    """
    Solve the linear system a*x = b
    Note that only the same subset of the rows and columns of a might be "warm"

    :param a: n x n matrix
    :param b: n x 1 vector

    :return: The solution vector x (which may contain NaNs
    """
    # make sure a is quadratic
    assert a.shape[0] == a.shape[1]
    # make sure the vector b has the right number of entries
    assert b.size == a.shape[0]

    x = np.nan * np.ones(b.size)
    v, mat = __valid(a)

    if v.any():
        x[v] = np.linalg.solve(mat, b[v])

    return x