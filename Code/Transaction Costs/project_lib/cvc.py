# # imports
from sklearn.isotonic import IsotonicRegression as IR
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import multiprocessing as mp
from project_lib.utils import *

def cov_calc(spectrum, V):
    """
    Takes (modified) spectrum and eigenvectors, returns a covariance matrix.
    """
    cov = V.dot(np.diag(spectrum)).dot(V.T)
    return 0.5 * (cov + cov.T)

def _cross_validate_eigenvalues(X, L, K, df, isotonic):
    N, T = X.shape
    Ycv = np.zeros(X.shape)

    # observations per fold
    obs_fold = T / np.float(K)

    # observations in small and observations in large K
    lowern = np.floor(obs_fold)
    uppern = np.ceil(obs_fold)
    lowern, uppern = np.int(lowern), np.int(uppern)

    # how many large and how many small K
    upperf = T - lowern * K
    lowerf = K - upperf

    np.random.seed(42)
    permut = np.random.permutation(T)

    for i in range(K):
        cv_idx1 = np.zeros([T], dtype=bool)

        # get the indices for the eigenvalue estimation
        if i < lowerf:
            cv_idx1[permut[i * lowern: (i + 1) * lowern]] = True
        else:
            cv_idx1[permut[lowerf * lowern + (i - lowerf) * uppern:
                           lowerf * lowern + (i - lowerf + 1) * uppern]] = True

        # get the other indices for the variance estimation
        cv_idx2 = ~cv_idx1

        E = X[:, cv_idx2].dot(X[:, cv_idx2].T)

        Lcv, Vcv = np.linalg.eigh(E)
        Ycv[:, cv_idx1] = Vcv.T.dot(X[:, cv_idx1])

    lcv = np.sum(Ycv ** 2, 1) / (T - df)

    # run isotonic regression
    if isotonic:
        lcv = IR().fit_transform(L, lcv)
        
    return lcv


def auxilliary_data(X, beta):
    T = X.shape[0]
    j = np.arange(0, T)
    weights = T * (1 - beta) * beta ** j / (1 - beta ** T)
    W = np.diag(weights[::-1])
    return W ** (1 / 2) @ X  


def CV(X, beta=None, K=10, df=0, scaling=True, isotonic=True):
    T, N = X.shape
    T = T - df
    
    if beta:
        X = auxilliary_data(X, beta)
    
    E = X.T @ X / T

    # eigenvalue decomposition of the original data
    L, V = np.linalg.eigh(E)

    # generate CV eigenvalues
    lcv = _cross_validate_eigenvalues(X.T, L, K, df, isotonic=isotonic)

    if scaling:
        scaling = E.trace() / np.sum(lcv)
        lcv = scaling * lcv
        assert math.isclose(E.trace(), np.sum(lcv), rel_tol=0.00001)

    Sigma = cov_calc(lcv, V)

    return Sigma, lcv


results = []
def average_result(x):
    results.append(np.sum(x ** 2, 1))

    
def wrapper_cv(X, L, K=10, df=0):
    T = X.shape[0]
    pool = mp.Pool(mp.cpu_count()-1)
    kf = KFold(n_splits=K, shuffle=True, random_state=123)
    for train_index, test_index in kf.split(X):
        fold = (train_index, test_index)
        pool.apply_async(cross_validate_eigenvalues_mp, args=(X, L, fold), callback=average_result)

    # Close the pool for new tasks
    pool.close()
    # Wait for all tasks to complete at this point
    pool.join()

    lcv = np.sum(results, 0) / (T - df)
    # run isotonic regression
    lcv = IR().fit_transform(L, lcv)
    return lcv


def cross_validate_eigenvalues_mp(X, L, K):
    X = X.T
    Ycv = np.zeros(X.shape)  # N x T
    E = X[:, K[0]].dot(X[:, K[0]].T)
    Lcv, Vcv = np.linalg.eigh(E)
    Ycv[:, K[1]] = Vcv.T.dot(X[:, K[1]])
    return Ycv


def CV_mp(X, beta=None, K=10, df=0, scaling=True):
    """
    Return a cross-validation based covariance estimate
    """
    T, N = X.shape
    
    if beta:
        X = auxilliary_data(X, beta)
    
    # sample covariance matrix
    E = X.T @ X / (T - df)

    # eigenvalue decomposition of the original data
    L, V = linalg.eigh(E)

    # generate CV eigenvalues
    lcv = wrapper_cv(X, L, K, df)

    if scaling:
        scaling = E.trace() / np.sum(lcv)
        lcv = scaling * lcv
        assert math.isclose(E.trace(), np.sum(lcv), rel_tol=0.00001)

    Sigma = cov_calc(lcv, V)

    return Sigma, lcv



def CV_new(X, beta=None, K=10, df=0, scaling=True, isotonic=True):
    T, N = X.shape
    T = T - df
    
    if beta:
        X = auxilliary_data(X, beta)
    
    m = CovFiltIntraday(X.T)
#     m = bahc.filterCovariance(X.T, Nboot=10)
    d, v = np.linalg.eigh(m)
    tmp = v.T @ np.diag(1 / np.sqrt(abs(d))) @ v
    tmp2 = v.T @ np.diag(np.sqrt(abs(d))) @ v
    X = X @ tmp
    E = X.T @ X / T

    # eigenvalue decomposition of the original data
    L, V = linalg.eigh(E)

    # generate CV eigenvalues
    lcv = _cross_validate_eigenvalues(X.T, L, K, df, isotonic=isotonic)

    Sigma = cov_calc(lcv, V)

    if scaling:
        scaling = E.trace() / np.sum(lcv)
        lcv = scaling * lcv
        assert math.isclose(E.trace(), np.sum(lcv), rel_tol=0.00001)
        
    return tmp2@Sigma@tmp2, lcv



def CovFiltIntraday(ret):
    import bahc
    import numpy as np
    
    C = np.cov(ret,bias=True)

    l,v = np.linalg.eigh(C)
    l[l<0] = 0.
    ind_lmax = np.argmax(l)
    
    #Factor loading Matrix
    P = np.dot(v,np.diag(np.sqrt(l)))
    
    #Estimated Factor Scores
    A = np.dot(np.dot(P.T,np.linalg.pinv(C)),ret)
 
    ret_partial = np.dot( np.delete(P,ind_lmax,axis=1), np.delete(A,ind_lmax,axis=0) )
    
    C_mode = np.outer(P[:,ind_lmax],P[:,ind_lmax])
    
    return SingleHC(ret_partial.T) + C_mode

