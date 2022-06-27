import numpy as np
import pandas as pd
from scipy.stats import gmean

from project_lib.utils import *

def func(ret):
    lookback=21
    k = np.arange(0,lookback+1)
    wts = 1/11-1/231*k
    wts = np.flip(wts[:-1])
    return -np.average(ret, weights=wts)


def reversion(ret, lookback=21):
    """Generate signal
    
    :param ret: DataFrame of returns matrix
    :param lookback: Lookback period in which the signal is computed
    :return: Lagged signal matrix
    """
    
    
    alpharev_unadj = ret.rolling(window=lookback).apply(lambda x: -sum(x[0:lookback])/lookback, raw=True)
#     alpharev_unadj = func(ret)
#     alpharev_unadj = ret.rolling(window=21).apply(func, raw=True)
    alpharev_fill = alpharev_unadj.fillna(0)
    return alpharev_fill.shift(1)


def momentum(ret, window_short=22, window_long=250, weight=1.9):
    signal_fast = ret.ewm(span=np.floor(window_short)).mean()
    signal_slow = ret.ewm(span=np.floor(window_long)).mean()
    signal = weight*signal_slow + (1-weight)* signal_fast
    return np.tanh(signal).shift(1)


# def momentum(ret):            
#     # implementation 2 
#     cret = (1 + ret).rolling(window=252, min_periods=1).apply(np.prod, raw=True)
#     cret -= 1
#     signal = cret.shift(21)
#     signal.iloc[:252] = np.nan
#     return signal.shift(1)


def norm(ret, lookback=1000):
    """Generate signal
    
    :param ret: DataFrame of returns matrix
    :param lookback: Lookback period in which the signal is computed
    :return: Lagged signal matrix
    """
    lookback = 1000
    alpha = 1-0.94   # This is ewma's decay factor.
    weights = list(reversed([(1-alpha)**n for n in range(lookback)]))
    ewma = partial(np.average, weights=weights)

    var_init = ret.var().to_frame().transpose()

    # join the initial variance with the remaining data
    data = pd.concat([var_init, ret[:-1] ** 2], ignore_index=True)
    data.index = ret.index

    vol = np.sqrt(data.rolling(lookback).apply(ewma, raw=True))
    
    # vol adjust
    ret_adjust = ret / vol  
    
    # clip vol-adjusted returns
#     ret_adjust = ret_adjust.clip(-clip, +clip)
    rets_adjust = winsorization(ret_adjust)
    
    return ret_adj, vol


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
    
    
def ts_norm(ret):
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
    return ret_adjust.values, vol.values


def cs_norm(rets, df_option=False):
    """Cross-sectionally normalize the returns
    
    :param ret: Returns where the time is row-wise
    """
    cross_sec_vol = np.array([rets.std(ddof=0, axis=1).values,] * rets.shape[1]).T
    rets_adjust = rets / cross_sec_vol
    rets_adjust = winsorization(rets_adjust)
    if df_option:
        return pd.DataFrame(rets_adjust, columns=rets.columns, index=rets.index), pd.DataFrame(cross_sec_vol, columns=rets.columns, index=rets.index)
    else:
        return rets_adjust, cross_sec_vol

    
def scale(signals, vol):
    euclid_norm = np.sqrt((signals * signals).sum(axis=1))

    # Divide each column of signal by the Euclidean norm
    cross_sec_vol = np.array([euclid_norm,] * signals.shape[1]).T
    signal_scaled = signals / cross_sec_vol
#     euclid_norm = np.sqrt((signals / vol * signals / vol).sum(axis=1))

    # Divide each column of signal by the Euclidean norm
#     signal_scaled = signals.fillna(0).apply(lambda x: x / euclid_norm, axis=0)
    return signal_scaled


def xs_normalization(df):
    """Demean, standardize and winsorize the alpha vector
    
    :param df: DataFrame of signals
    :return: Processed signals
    """
    cross_sec_mean = np.array([df.mean(axis=1),] * df.shape[1]).T
    normalized_df = (df - cross_sec_mean) 
    cross_sec_vol = np.array([normalized_df.std(ddof=1, axis=1),] * df.shape[1]).T
    normalized_df /= cross_sec_vol
    
#     normalized_df /= np.array([np.linalg.norm(normalized_df, ord=1, axis=1),] * df.shape[1]).T
#     normalized_df = pd.DataFrame(normalized_df, index=df.index, columns=df.columns)
    normalized_df[normalized_df>3] = 3
    normalized_df[normalized_df<-3] = -3
#     normalized_df /= np.array([np.linalg.norm(normalized_df, ord=2, axis=1),] * df.shape[1]).T

    
#     cross_sec_mean = np.array([df.mean(axis=1),] * df.shape[1]).T
#     normalized_df = (df - cross_sec_mean) 

#     N = df.shape[1]
#     rk_signal = (pd.DataFrame(df).rank(axis=1) - 1)
#     x_rk = rk_signal / (N+1) 
#     x_rk -= np.mean(x_rk )
#     x_rk /= np.array([np.linalg.norm(x_rk, ord=1,axis=1), ] * N).T
#     normalized_df = x_rk.values
    return normalized_df #, pd.DataFrame(cross_sec_vol, index=df.index, columns=df.columns)


def mom_return(r, window=252, exclude_window=21):
    """
    The momentum signal of a given stock is computed as the geometric average of the previous 252 returns on that
    stock but excluding the most recent 21 returns, that is, the geometric average over the previous year but
    excluding the previous month
    """
    n = r.shape[0]
    glob_window = window + exclude_window

    if n < glob_window:
        print(n)
        raise ValueError("Length of returns is not equal")

    m_wind = r[-glob_window:-exclude_window]
    m_cumwind = m_wind + 1
    er = np.apply_along_axis(gmean, 0, m_cumwind) - 1
    return er


# compute the oscillator
def osc(prices, fast=32, slow=96, scaling=True):
    """Compute momentum signal based on Baz et al. (2015) 
    :param prices: Prices where the time is row-wise
    :return: DataFrame of calculated signals
    """
    
    f,g = 1 - 1/fast, 1-1/slow
    if scaling:
        s = np.sqrt(1.0 / (1 - f * f) - 2.0 / (1 - f * g) + 1.0 / (1 - g * g))
    else:
        s = 1
        
    return (prices.ewm(com=fast-1).mean() - prices.ewm(com=slow-1).mean()) / s

