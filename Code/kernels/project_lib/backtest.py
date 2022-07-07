# backtest functions
from scipy import stats
import numpy as np
import pandas as pd
from project_lib.NLW import *
from project_lib.cvc import *

####################### Signal construction ################################


def winsorization(a):
    N = a.shape[1]
    a_mean = stats.trim_mean(a, 0.1, axis=1)
    a_mad = stats.median_abs_deviation(a, axis=1)
    a_lo = a_mean - 5 * a_mad
    a_hi = a_mean + 5 * a_mad
    a_hi = np.array([a_hi] * N).T
    a_lo = np.array([a_lo] * N).T
    cond1 = a > a_hi
    cond2 = a < a_lo
    out = np.where(cond2, a_lo, np.where(cond1, a_hi, a))
    return pd.DataFrame(out, columns=a.columns, index=a.index)


def reversion(ret, lookback=21):
    """Generate signal

    :param ret: DataFrame of returns matrix
    :param lookback: Lookback period in which the signal is computed
    :return: Lagged signal matrix
    """

    alpharev_unadj = ret.rolling(window=lookback).apply(
        lambda x: -sum(x[0:lookback])/lookback, raw=True)
    alpharev_fill = alpharev_unadj.fillna(0)
    return alpharev_fill.shift(1)


def cross_sectional_norm(rets):
    """Cross-sectionally normalize the returns

    :param ret: Returns where the time is row-wise
    """
    cross_sec_vol = np.array(
        [rets.std(ddof=0, axis=1).values, ] * rets.shape[1]).T
    rets_adjust = rets / cross_sec_vol
#     rets_adjust = winsorization(rets_adjust)
    twosig = rets_adjust.stack().std() * 2
    rets_adjust[rets_adjust > twosig] = twosig
    return rets_adjust, pd.DataFrame(cross_sec_vol, columns=rets.columns, index=rets.index)


def xs_normalization(df):
    """Demean, standardize and winsorize the alpha vector

    :param df: DataFrame of signals
    :return: Processed signals
    """
    cross_sec_mean = np.array([df.mean(axis=1).values, ] * df.shape[1]).T
    normalized_df = (df - cross_sec_mean)
    cross_sec_vol = np.array(
        [normalized_df.std(ddof=0, axis=1).values, ] * df.shape[1]).T
#     cross_sec_vol = np.array([abs(normalized_df).sum(axis=1).values,] * df.shape[1]).T
    normalized_df /= cross_sec_vol
    normalized_df[normalized_df > 3] = 3
    normalized_df[normalized_df < -3] = -3
    normalized_df = normalized_df.fillna(0)
    return normalized_df, pd.DataFrame(cross_sec_vol, index=df.index, columns=df.columns)


def gaussian_rff(X, R):
    """
    Kernel approximation using random Fourier features. Based on "Random Features
    for Large-Scale Kernel Machines" by Rahimi and Recht (2007)

    input : 
        X - the original data matrix     [n x m]
        R - the desired output dimension [1 x 1]
    output : 
        ZZ - kernel matrix [R x R]
    """
    D = X.shape[1]
    W = np.random.normal(loc=0, scale=1, size=(R, D))
    b = np.random.uniform(0, 2*np.pi, size=R)
    B = np.repeat(b[:, np.newaxis], X.shape[0], axis=1)
    #D = X.shape[1]
    norm = 1. / np.sqrt(R)
    Z = norm * np.sqrt(2) * np.cos(W @ X.T + B)
    ZZ = Z.T@Z
    return ZZ

###################### Covariance models ############################


def construct_cov(d, v):
    """Reconstruct covariance matrix from eigenvalues and inverted eigenvectors of covariance matrix

    :param d: Eigenvalues of covariance matrix
    :param v: Eigenvectors of covariance matrix
    :return: Reconstructed covariance matrix
    """
    v_tpose = np.linalg.inv(v)
    d_mat = np.diag(d)
    return v.dot(d_mat.dot(v_tpose))


def get_cov(data, factor=None, method='sample', square_root=True, df=0):
    """Obtain estimated covariance

    :param ret: Returns matrix
    :param method: Cleaning method
    :param df: Degrees of freedom correction
    :return: Reconstructed inverse square root covariance matrix
    """

    T, N = data.shape

#     assert data.shape[0] > data.shape[1]

    cov = data.T @ data / T

#     corr, std = cov2corr(cov)

    d, v = np.linalg.eigh(cov)

    if method == 'kernel':
        cov2 = gaussian_rff(data.T, min(N, T))
        d, v = np.linalg.eigh(cov2)

    elif method == 'sample':
        pass

    elif method == 'nls':
        _, d = NLSHRINK(data.T)

    elif method == 'ls':
        LW = LedoitWolf().fit(data)
        coef = LW.shrinkage_
        d = (1-coef) * d + (coef) * np.mean(d)

    elif method == 'id':
        d = np.ones(N) * np.mean(d)

    else:

        raise NotImplementedError(
            "Return covariance {} not implemented".format(method))

    if square_root:
        return construct_cov(np.sqrt(d), v)

    else:
        return construct_cov(d, v)

########## cross-correlation matrix #########################


def get_xscorr(corr_RX, method='id'):
    """Obtain interpolated cross-correlation matrix

    :param corr_RX: cross-correlation matrix
    :return: interpolated cross-correlation matrix
    """

    N, M = corr_RX.shape

    if method == 'id':
        if M > N:
            # target is stacked identity matrix. Care: the dimensionality of this matrix is HARD-CODED
            dim = 2
            corr_RX = 1/2 * np.tile(np.eye(N), dim)

        else:
            # target is identity matrix
            corr_RX = np.eye(N)

    elif method == 'sample':
        pass

    return corr_RX
