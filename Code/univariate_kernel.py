import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from localreg import *
import multiprocessing as mp

############ supporting functions #########################
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

############## univariate kernel functions #####################################
# used in nonparametric cca, 2016, Livescu, Michaeli & Wang
def kernel_regression(X, R, kernel, col):
    """
        function to calculate univariate kernel regression of asset on signal
        source : https://github.com/sigvaldm/localreg
        inputs:
            X      : matrix of signals [m x l]
            R      : matrix of returns [m x l]
            kernel : kernel type   [string] -> rbf.triangular, rbf.gaussian, rbf.epanechnikov
            col    : column for which to calculate regression [1 x 1]
        output:
            numpy array of kernel regression result [1 x l]

    """
    output = localreg(np.array(X[:, col]), np.array(R[:, col]),
                      x0=None, degree=0, kernel=kernel, radius=1, frac=None)

    return output

def nw_regression(X, R, col):
    """
        Nadaraya-Watson regression (exact implementation of nonparametric cca, 2016, Livescu, Michaeli & Wang)
        
        inputs: X : matrix of signals [m x l]
                R : matrix of returns [m x l]
                col : column to perform regression on [1 x 1]
        outputs:
                Univariate Nadaraya-Watson regression of return on signal [1 x l]
    """
    kr = KernelReg(np.array(R[:, col]), np.array(X[:, col]), var_type='c')
    estimator = kr.fit()
    return estimator[0]

def parallel_kernel(X, R, kernel):
    """
        function to calculate kernel regressions of each asset in a parallelized manner
        
        inputs:
            X : matrix of signals [m x l]
            R : matrix of returns [m x l]
            kernel : kernel type  [string] -> rbf.triangular, rbf.gaussian, rbf.epanechnikov
        output:
            numpy array of univariate kernel regressions [m x l]
    """
    # initialize pool
    pool = mp.Pool(mp.cpu_count())
    
    # parallelized kernel regression of signals
    results = pool.starmap(kernel_regression, [(
        X, R, kernel, a) for a in np.arange(0, R.shape[1], 1)])
    
    # alternative : Nadaraya-Watson
    #results = pool.starmap(nw_regression, [(
    #    X, R, a) for a in np.arange(0, R.shape[1], 1)])
    
    results = np.array(results).T
    pool.close()
    pool.join()
    return results    
    
######## backtest function for univariate kernel #######################

def simulate_backtest(cov_models, returns, signals,
                      n_lag=0, Tcov=1000, rebalance='1B', option='unconstrain', option2='new',univariate_kernel=rbf.gaussian):
    """Run a backtest on return data with specified rebalance periods and model parameters
    
    :param cov_models: List of covariance models
    :param xscov_models: String of cross-correlation model
    :param returns: Dataframe of returns where the time component is row-wise
    :param signals: Dataframe of signals where the time component is row-wise
    :param n_lag: Number of day to lag
    :param Tcov: Time length used to compute the covariance and cross-correlation matrices
    :param rebalance: Pandas resample frequency
    :return: Optimal portfolio weights
    """
    
    print('OPTIMIZING FOR MODELS: {}'.format('-'.join(cov_models)))
    
    all_dates = returns.index.tolist()
    T_tot = returns.shape[0]
    
    #use pandas resample method to obtain dates of rebalance
    rebalance_dates = returns.resample(rebalance).asfreq().index
    
    #dictionary of dataframes to store the calculated portfolio weights
    portfolio_weights = {m: pd.DataFrame(columns=returns.columns) for m in cov_models}
    
    #begin only when a covariance matrix can be computed on specified interval + lag
    start_time = time.time()
    for i in range(Tcov + n_lag, T_tot):
        #only rebalance on the specified dates
        if all_dates[i] in rebalance_dates:
            start, end = i - Tcov - n_lag, i - n_lag 
            date_start, date_end, date_rebal = all_dates[start], all_dates[end-1], all_dates[i]
            
            print('Rebalance: {}, Covariance Matrix: {} - {}'.format(
                date_rebal.strftime('%b %d, %Y'), date_start.strftime('%b %d, %Y'), date_end.strftime('%b %d, %Y')))
                
            # signal option
            R = returns.values[start:end]
            X = signals.values[start:end]
            X_end = signals.values[end]
            
            N = R.shape[1]
                
            # loop over different covariance models
            for m in cov_models:
                
                    # get univariate kernel weights
                    results = parallel_kernel(X[(i-Tcov):i,:],
                                              R[(i-Tcov):i,:],
                                              univariate_kernel)

                    cov_R_half = get_cov(R[(i-Tcov):i,:], method=m, square_root=True)
                    
                    K_pl =(results.T @ results)

                    # make sure it's sorted
                    eigen_val, eigen_vec = np.linalg.eig(K_pl)
                    order = np.argsort(eigen_val)[::-1]
                    idx = np.empty_like(order)
                    idx[order] = np.arange(len(order))
                    eigen_vec[:] = eigen_vec[:, idx] 
                    
                    w = eigen_vec.T @ np.linalg.inv(cov_R_half)
                    
                    # aggregate all canonical portfolios into one by weighing each by its canonical correlation
                    w = eigen_val @ w.T
                        
                    # save the weights for each model
                    portfolio_weights[m].loc[date_rebal] = w
    
    print("Time: %s seconds" % str(round(time.time() - start_time)))
    return portfolio_weights


