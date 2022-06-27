# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:07:23 2020

@author: Vincent
"""
import numpy as np
from cvxopt import solvers, matrix
import cvxpy
from cvxpy import constraints
from project_lib.utils import *

from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import gmean

# Install cvxpy with this command:
# conda install -c conda-forge cvxpy

# I used a variant of the formula to optimize, since it looks
# like there's an error in the one from the paper.
# https://kyle-stahl-mn.com/stock-portfolio-optimization


def mom_return(r, window=252, exclude_window=21):
    """
    The momentum signal of a given stock is computed as the geometric average of the previous 252 returns on that
    stock but excluding the most recent 21 returns, that is, the geometric average over the previous year but
    excluding the previous month
    """
    n = r.shape[0]
    glob_window = window + exclude_window

    if n < glob_window:
        raise ValueError("Length of returns is not equal")

    m_wind = r[-glob_window:-exclude_window, :]
    m_cumwind = m_wind + 1
    er = np.apply_along_axis(gmean, 0, m_cumwind) - 1
    return er


def mvm_wturnover(r, er, w_0, est_cov, k=10 ** -3):
    T, N = r.shape

    # Compute target return from top 5% percentile
    # The target return b is computed as the arithmetic average of the momentums of the stocks belonging to the
    # top-quintile stocks according to momentum.
    top5perc = np.percentile(er, 95)
    b = er[er >= top5perc].mean()
    S = est_cov

    # Optimization
    w = cvxpy.Variable([N, 1])
    devs = cvxpy.abs(w - w_0)
    constraints = []
    constraints.append(er.T * w == b)
    constraints.append(sum(w) == 1)
    obj = cvxpy.Minimize(cvxpy.atoms.quad_form(w, S) + [k] * sum(devs))
    problem = cvxpy.Problem(obj, constraints)
    problem.solve(solver='CVXOPT', verbose=True)
    return w.value


def ewtq(er):
    N = er.shape[0]
    w = np.zeros(N)
    top5perc = np.percentile(er, 80)
    n_top = 1/np.sum(er >= top5perc)
    w[er >= top5perc] = er[er >= top5perc]/n_top
    return w


def mvm(er, est_cov):
    N = est_cov.shape[0]

    # Compute target return from top 20% percentile. The target return b is computed as the arithmetic average of the
    # momentums of the stocks belonging to the top-quintile stocks according to momentum.
    top5perc = np.percentile(er, 80)
    b = er[er >= top5perc].mean()
    S = est_cov

    i = np.ones((N, 1))
    m = er.reshape(N, 1)
    C = np.dot(np.dot(i.T, np.linalg.inv(S)), i)
    D = np.dot(np.dot(m.T, np.linalg.inv(S)), i)
    E = np.dot(np.dot(m.T, np.linalg.inv(S)), m)

    num = m * (C * b - D) + i * (E - D * b)
    den = E * C - D ** 2

    w = np.dot(np.linalg.inv(S), (num / den))
    return w


def gmv(est_cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    N = est_cov.shape[0]
    prec = np.linalg.inv(est_cov)
    denom = np.matmul(np.matmul(np.ones(N), prec), np.ones(N))
    return np.matmul(prec, np.ones(N)) / denom


def gmv_wturnover(w_0, est_cov, k=10 ** -3):
    # Optimization
    N = est_cov.shape[0]
    w = cvxpy.Variable([N, 1])
    devs = cvxpy.abs(w - w_0)
    constraints = []
    constraints.append(sum(w) == 1)
    obj = cvxpy.Minimize(cvxpy.atoms.quad_form(w, est_cov) + [k] * sum(devs))
    problem = cvxpy.Problem(obj, constraints)
    problem.solve(solver='CVXOPT', verbose=True)
    return w.value.T


def computePortfolios_turnover(r, est_cov, w_minvar_old, w_meanvar_old, kappa):
    w_minvar = gmv_wturnover(w_minvar_old, est_cov, kappa)
    er = mom_return(r, window=252, exclude_window=21)
    w_meanvar = mvm_wturnover(r, er, w_meanvar_old, est_cov, kappa)
    return w_minvar, w_meanvar


def compute_gmv_turnover(est_cov, w_minvar_old, kappa):
    w_minvar = gmv_wturnover(w_minvar_old, est_cov, kappa)
    return w_minvar


def compute_mvm_turnover(est_cov, r, w_meanvar_old, kappa):
    er = mom_return(r, window=252, exclude_window=21)
    w_meanvar = mvm_wturnover(r, er, w_meanvar_old, est_cov, kappa)
    return w_meanvar


def EWQU(er):
    pct = pd.DataFrame(er).rank(pct=True)
    long = (pct > 0.8) *1
    long = long.divide(long.abs().sum(),0) 
    short = (pct < 0.2)*(-1.0)
    short = short.divide(short.abs().sum(),0) 
    w = short.add(long, fill_value=0).fillna(0)
    w = w.values.squeeze()
    return w


def mvm(er, cov):
    """Markowitz with signal
    
    :param er: Expected returns
    :param cov: Covariance matrix
    :return: Optimal weight vector
    """
    N = cov.shape[0]
    
    # Compute target return from top 20% percentile. 
    # The target return b is computed as the arithmetic average of the momentums 
    # of the stocks belonging to the top-quintile stocks according to momentum.
    top5perc = np.percentile(er, 80)
    b = er[er >= top5perc].mean()
#     w = EWQU(er)
#     b = np.dot(er,w)
    
    S = cov
    
    i = np.ones(N)
    m = er
    A = np.dot(np.dot(i, np.linalg.inv(S)), i)
    B = np.dot(np.dot(m, np.linalg.inv(S)), i)
    C = np.dot(np.dot(m, np.linalg.inv(S)), m)
    
    num = i * (C - b * B) + m * (b * A - B) 
    den = A * C - B ** 2
    w = np.dot(np.linalg.inv(S), (num / den))
    return w


def compute_ptf(X_end, cov_R, cov_X, cov_RX, option='default'):
    N = cov_R.shape[1]
    # unconstrained optimal policy matrix 
    A = np.linalg.inv(cov_X) @ cov_RX.T @ np.linalg.inv(cov_R) 
    
    if option == 'default':  
        w = A.T @ X_end 
#         gain = np.trace(cov_RX) / np.linalg.norm(X_end, ord=2) 
        gain = np.trace(cov_RX) / N
        lambd =  gain / np.trace(A.T @ cov_RX.T)
        w = w * lambd
        
    elif option == 'mkz_default':
        A = np.linalg.inv(cov_R)
        w = A.T @ X_end 
#         gain = np.trace(cov_X) / np.linalg.norm(X_end, ord=2)  
        gain = np.trace(cov_X) / N
        lambd =  gain / np.trace(A.T @ cov_X.T)
        w = w * lambd 

    elif option == 'mkz_default_l2':
        w = solve(cov_R, X_end) / (inv_a_norm(X_end, cov_R) ** 2)
        w /= np.linalg.norm(w, ord=2)
        
    elif option == 'sum0_l1':
        gain = 1
        b = np.sum(A.T @ X_end)
        w_gmv = solve(cov_R, np.ones(N)) / (inv_a_norm(np.ones(N), cov_R) ** 2)
        w = A.T @ X_end - w_gmv * b
        w /= np.linalg.norm(w, ord=1) * 2

    elif option == 'sum1':
        # useful constants
#         X_end /=inv_a_norm(X_end, cov_X) 
        a = (inv_a_norm(np.ones(N), cov_R) ** 2) * (inv_a_norm(X_end, cov_X) ** 2)
        b = np.sum(A.T @ X_end)
        c = np.trace(A.T @ cov_RX.T)
        w_tangent = A.T @ X_end / b

#                     w = EWQU(X_end)
#                     gain = np.dot(X_end,w)
#                     gain = X_end[X_end >= top5perc].mean()

#         top5perc = np.percentile(X_end, 80)
#         gain = X_end[X_end >= top5perc].mean()
        
#         gain = np.trace(cov_RX) / np.linalg.norm(X_end, ord=2)
        gain = np.trace(cov_RX) / N
        lambd =  gain / np.trace(A.T * cov_RX.T)
        lambd = (gain * a * b - b ** 2) / (a * c - b ** 2)

        # compute gmv portfolio
        w_gmv = solve(cov_R, np.ones(N)) / (inv_a_norm(np.ones(N), cov_R) ** 2)
        w = lambd* w_tangent + (1-lambd) * w_gmv

    elif option == 'gmv':
        # compute gmv portfolio
        w_gmv = solve(cov_R, np.ones(N)) / (inv_a_norm(np.ones(N), cov_R) ** 2)
        w = w_gmv

    elif option == 'mkz_sum1':
        A = np.linalg.inv(cov_R)
        
        a = (inv_a_norm(np.ones(N), cov_R) ** 2) * (inv_a_norm(X_end, cov_X) ** 2)
        print(np.sum(X_end))
        b = np.sum(A.T @ X_end)
        c = np.trace(A.T @ cov_X)
        w_tangent = A.T @ X_end / b

#                     w = EWQU(X_end)
#                     gain = np.dot(X_end,w)
#                     gain = X_end[X_end >= top5perc].mean()

#         top5perc = np.percentile(X_end, 80)
#         gain = X_end[X_end >= top5perc].mean()
#         gain = np.trace(cov_X) / np.linalg.norm(X_end, ord=2)
        gain = np.trace(cov_X) / N
    
        lambd = (gain * a * b - b ** 2) / (a * c - b ** 2)

        # compute gmv portfolio
        w_gmv = solve(cov_R, np.ones(N)) / (inv_a_norm(np.ones(N), cov_R) ** 2)
#         print(w_tangent)
        w = lambd* w_tangent + (1-lambd) * w_gmv
    
    elif option == 'sum0':
        b = np.dot(np.ones(N),A.T @ X_end)
        c = np.trace(A.T @ cov_RX.T)
        a = (inv_a_norm(np.ones(N), cov_R) ** 2) * (inv_a_norm(X_end, cov_X) ** 2)
        
        gain = np.trace(cov_RX) / N
#         gain = np.trace(cov_X) / np.linalg.norm(X_end, ord=2)
        
        scaling = gain / (c/b - b /a)
        
        w_tangent = A.T @ X_end / b
        w_gmv = solve(cov_R, np.ones(N)) * (inv_a_norm(X_end, cov_X) ** 2) / a
        w = (w_tangent - w_gmv)
        w *= scaling 
        
    elif option == 'mkz_sum0':
        # compute gmv portfolio
        A = np.linalg.inv(cov_R)
        b = np.dot(np.ones(N), A.T @ X_end)
        c = np.trace(A.T @ cov_X)
        a = (inv_a_norm(np.ones(N), cov_R) ** 2) * (inv_a_norm(X_end, cov_X) ** 2)
        gain = np.trace(cov_X) / N
#         gain = np.trace(cov_X) / np.linalg.norm(X_end, ord=2)
        
        scaling = gain / (c/b - b /a)
        
        w_tangent = A.T @ X_end / b
        w_gmv = solve(cov_R, np.ones(N)) * (inv_a_norm(X_end, cov_X) ** 2) / a
        w = (w_tangent - w_gmv)
        w *= scaling 

    elif option == 'ew':
#         avg_X = np.trace(cov_X)  / N * np.eye(N)
#         avg_R = np.trace(cov_R)  / N * np.eye(N)
#         avg_RX = np.trace(cov_RX)  / N * np.eye(N)
#         A = np.linalg.inv(avg_X) @ avg_RX.T @ np.linalg.inv(avg_R)
#         w = A.T @ X_end 
#         gain = np.trace(avg_RX) / np.linalg.norm(X_end, ord=2)
#         lambd =  gain / np.trace(A.T @ avg_RX.T)
#         w = lambd * w  
        w = X_end / N 
#         print(w)

    else:
        raise NotImplementedError("Return portfolio weights {} not implemented".format(option))

    return w.flatten()


