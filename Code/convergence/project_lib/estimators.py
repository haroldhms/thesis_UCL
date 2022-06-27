import numpy as np

# import cov cleaning methods 
import pandas as pd
from sklearn.covariance import LedoitWolf
from project_lib.cv_svd import *
from project_lib.arrr import *
from project_lib.NLW import *
from project_lib.hcal import *
from project_lib.cvc import *
from project_lib.svd_rie import *
from project_lib.utils import *
import bahc

from sklearn.covariance import LedoitWolf
np.warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from arch.univariate.mean import ZeroMean
from arch.univariate.volatility import EWMAVariance
from scipy import stats
from scipy.linalg import toeplitz


def cross_sectional_norm(rets):
    """Cross-sectionally normalize the returns

    :param ret: Returns where the time is row-wise
    """
    
    xs_vol = np.array([rets.std(ddof=0, axis=1),] * rets.shape[1]).T
    rets_adjust = rets / xs_vol
#     twosig = np.std(rets_adjust, ddof=1) * 2
#     rets_adjust[rets_adjust > twosig] = twosig
    rets_adjust = winsorization(rets_adjust)
    return rets_adjust, xs_vol[-1, :]


def igarch(ret):
    """Compute IGARCH(1,1) (aka J.P Morgan RiskMetrics 1994 model)
    :param ret: Log-returns where the time is row-wise
    :return: volatility
    """
    # compute initial variance
#     var_init = ret.var().to_frame().transpose()
    var_init = ret.var().to_frame().transpose()
    
    # join the initial variance with the remaining data
    data = pd.concat([var_init, ret[:-1] ** 2], ignore_index=True)
    data.index = ret.index
    
    # compute volatility using Pandas ewm
    vol = np.sqrt(data.ewm(alpha=(1-0.94), adjust=False).mean())
    
    return vol
    
    
def time_series_norm(ret):
    """Devolatize Returns with vol estimated with a RiskMetrics 1994 model
    :param prices: Prices where the time s row-wise
    :return: DataFrame of vol-scaled returns and volatility
    """
    
    # compute vol
    vol = igarch(pd.DataFrame(ret))
    
    # vol adjust
    ret_adjust = ret / vol  
    
    # clip vol-adjusted returns
    rets_adjust = winsorization(ret_adjust)
    return ret_adjust.values, vol.values[-1]


def SCM_wrapper(X, norm=False):
    if norm:
        X, std = time_series_norm(X)
#         X, std = cross_sectional_norm(X)
    else: 
        X, std = X, np.ones(X.shape[1])
    Sigma_tmp = X.T@X/X.shape[0]
    Sigma_scm = corr2cov(Sigma_tmp, std)
    return Sigma_scm


def LS_wrapper(X, norm=False):
    if norm:
        X, std = time_series_norm(X)
#         X, std = cross_sectional_norm(X)
    else: 
        X, std = X, np.ones(X.shape[1])
    Sigma_tmp = LedoitWolf().fit(X).covariance_
    Sigma_ls = corr2cov(Sigma_tmp, std)
    return Sigma_ls


def NLS_wrapper(X, norm=False):
    if norm:
        X, std = time_series_norm(X)
#         X, std = cross_sectional_norm(X)
    else: 
        X, std = X, np.ones(X.shape[1])
    Sigma_tmp = NLSHRINK(X.T)[0]
    Sigma_nl = corr2cov(Sigma_tmp, std)
    return Sigma_nl


def CV_wrapper(X, norm=True):
    if norm:
        X, std = time_series_norm(X)
#         X, std = cross_sectional_norm(X)
        std = np.sqrt(np.diag(X.T@X/X.shape[0]))
        X = X / std 
    else: 
        X, std = X, np.ones(X.shape[1])
    Sigma_tmp = CV(X)[0]
    Sigma_cv = corr2cov(Sigma_tmp, std)
    
    return Sigma_cv



def prototype_wrapper(X, market=False, norm=False):
    T, N = X.shape
    
    if market:
        cov_bahc = pca_nls(X)
    else: 
        cov_bahc = SingleHC(X, clip=True)
    
    # eigenvalue decomposition of BAHC matrix
    L, V = np.linalg.eigh(cov_bahc)
    
    # get square root of BAHC cov matrix
    cov_bahc_half = cov_calc(np.sqrt(L), V)
    
    # decorrelate the data matrix using the BAHC matrix
    Xtilde = X @ np.linalg.inv(cov_bahc_half)
    
    if norm:
        Xtilde, std = time_series_norm(Xtilde)
    else: 
        Xtilde, std = Xtilde, np.ones(Xtilde.shape[1])
    
    # apply CV on decorrelated data matrix
    cov_CV = CV(Xtilde)[0]
    cov_CV = corr2cov(cov_CV, std)
    
    # rebuild the final covariance using the BAHC matrix
    cov_final = cov_bahc_half @ cov_CV @cov_bahc_half
#     cov_final = corr2cov(cov_final, std)
    
    return cov_final


def ID_wrapper(X):
    return np.identity(X.shape[1]) / X.shape[1]
