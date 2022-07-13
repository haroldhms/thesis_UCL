import numpy as np
import pandas as pd
from collections import OrderedDict
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


def drawdown(series) -> pd.Series:
    """
    Compute the drawdown for a price series. The drawdown is defined as 1 - price/highwatermark.
    The highwatermark at time t is the highest price that has been achieved before or at time t.

    Args:
        series:

    Returns: Drawdown as a pandas series
    """
    return _Drawdown(series).drawdown


def monthlytable(nav) -> pd.DataFrame:
    """
    Get a table of monthly returns

    :param nav:

    :return:
    """
    def _cumulate(x):
        return (1 + x).prod() - 1.0

    r = nav.pct_change().dropna()
    # Works better in the first month
    # Compute all the intramonth-returns, instead of reapplying some monthly resampling of the NAV
    return_monthly = r.groupby([r.index.year, r.index.month]).apply(_cumulate)
    frame = return_monthly.unstack(level=1).rename(columns=lambda x: calendar.month_abbr[x])
    ytd = frame.apply(_cumulate, axis=1)
    frame["STDev"] = np.sqrt(12) * frame.std(axis=1)
    # make sure that you don't include the column for the STDev in your computation
    frame["YTD"] = ytd
    frame.index.name = "Year"
    frame.columns.name = None
    # most recent years on top
    return frame.iloc[::-1]


def performance(nav):
    return NavSeries(nav).summary()


class NavSeries(pd.Series):
    def __init__(self, *args, **kwargs):
        super(NavSeries, self).__init__(*args, **kwargs)
        if not self.empty:
            # check that all indices are increasing
            assert self.index.is_monotonic_increasing
            # make sure all entries non-negative
            assert not (self < 0).any(), "Problem with data:\n{x}".format(x=self.series)

    @property
    def series(self) -> pd.Series:
        return pd.Series({t: v for t, v in self.items()})

    @property
    def periods_per_year(self):
        if len(self.index) >= 2:
            x = pd.Series(data=self.index)
            return np.round(365 * 24 * 60 * 60 / x.diff().mean().total_seconds(), decimals=0)
        else:
            return 256

    def annualized_volatility(self, periods=None):
        t = periods or self.periods_per_year
        return np.sqrt(t) * self.dropna().pct_change().std()

    @staticmethod
    def __gmean(a):
        # geometric mean A
        # Prod [a_i] == A^n
        # Apply log on both sides
        # Sum [log a_i] = n log A
        # => A = exp(Sum [log a_i] // n)
        return np.exp(np.mean(np.log(a)))


    @property
    def returns(self) -> pd.Series:
        return self.pct_change().dropna()

    @property
    def __cum_return(self):
        return (1.0 + self.returns).prod() - 1.0

    def sharpe_ratio(self, periods=None, r_f=0):
        return (self.annual_r - r_f) / self.annualized_volatility(periods)

    @property
    def annual(self):
        return NavSeries(self.resample("A").last())

    @property
    def annual_r(self):
        r = NavSeries(self.resample("A").last()).pct_change()
        return self.__gmean(r + 1) - 1.0

    def summary(self, r_f=0):
        periods = self.periods_per_year

        d = OrderedDict()
        
        d["Return"] = 100 * self.__cum_return
        d["Annua Return"] = 100 * self.annual_r
        d["Annua Volatility"] = 100 * self.annualized_volatility(periods=periods)
        d["Annua Sharpe Ratio (r_f = {0})".format(r_f)] = self.sharpe_ratio(periods=periods, r_f=r_f)
        d["Max Drawdown"] = 100 * drawdown(self).max()
        d["Kurtosis"] = self.pct_change().kurtosis()
        
        x = pd.Series(d)
        x.index.name = "Performance number"
        
        return x


class _Drawdown(object):
    def __init__(self, series: pd.Series) -> object:
        """
        Drawdown for a given series
        :param series: pandas Series
        :param eps: a day is down day if the drawdown (positive) is larger than eps
        """
        # check series is indeed a series
        assert isinstance(series, pd.Series)
        # check that all indices are increasing
        assert series.index.is_monotonic_increasing
        # make sure all entries non-negative
        assert not (series < 0).any()

        self.__series = series

    @property
    def highwatermark(self) -> pd.Series:
        return self.__series.expanding(min_periods=1).max()

    @property
    def drawdown(self) -> pd.Series:
        return 1 - self.__series / self.highwatermark


def build_table(cov_models, pnl_results):
    order_of_models = cov_models
    full_table = pd.DataFrame(index=order_of_models)
    for m in pnl_results:
        perf = pnl_results[m].apply(performance)
        perf = perf.loc[["Annua Return", "Annua Volatility", "Annua Sharpe Ratio (r_f = 0)", "Max Drawdown", "Return", "Kurtosis"]]
        perf = perf.applymap(lambda x: float(x))
        full_table.loc[m, 'ER']  = round(perf['NAV']['Annua Return'], 2)
        full_table.loc[m, 'SD']  = round(perf['NAV']['Annua Volatility'], 2)
        full_table.loc[m, 'IR']  = round(perf['NAV']['Annua Sharpe Ratio (r_f = 0)'], 2)
        full_table.loc[m, 'MaxDD']  = round(perf['NAV']['Max Drawdown'], 2)
        full_table.loc[m, 'Return']  = round(perf['NAV']['Return'], 2)
        full_table.loc[m, 'Kurtosis']  = round(perf['NAV']['Kurtosis'], 2)
    return full_table


def ptf_summary(ptf_ret):
    num_days_year = 252
    d = OrderedDict()
        
#     d["AV"] = 100 * ptf_ret.mean() * 252
#     d["SD"] = 100 * ptf_ret.std() * np.sqrt(252)
#     d["IR"] =  d["AV"] /  d["SD"]
    d["AV"] = annual_return(ptf_ret) * 100
    d["SD"] = annual_volatility(ptf_ret) * 100
    d["IR"] = sharpe_ratio(ptf_ret)
    d["VaR"] = value_at_risk(ptf_ret)
    d["MDD"] = max_drawdown(ptf_ret)
    d["P2T"] = max_drawdown_length(ptf_ret)['peak_to_trough_maxdd']
    d["P2P"] = max_drawdown_length(ptf_ret)['peak_to_peak_maxdd']
    d["P2PL"] = max_drawdown_length(ptf_ret)['peak_to_peak_longest']
    d["Calmar"] = calmar_ratio(ptf_ret)
    d["Stability"] = stability_of_timeseries(ptf_ret)
    d["Omega"] = omega_ratio(ptf_ret)
    d["Sortino"] = sortino_ratio(ptf_ret)
    d["TailRatio"] = tail_ratio(ptf_ret)
    d["CSR"] = common_sense_ratio(ptf_ret)
    d["Kurtosis"] = ptf_ret.kurtosis()

    x = pd.Series(d)
    x.index.name = "Performance"
    return x

def weight_summary(w, ptf_ret):
    d = OrderedDict()
    
    c = 5/10000
    ptf_ret_net = (((1-c * w.diff().abs().iloc[1:].sum(axis=1))*(1+ptf_ret['Profit'].iloc[1:]))-1)*100
#     ptf_ret_net = ptf_ret['Profit'].iloc[1:] - c * w.diff().abs().iloc[1:].sum(axis=1) 
    d["IR_net"] = sharpe_ratio(ptf_ret_net)
    
    d["TO"] = w.diff().abs().iloc[1:].sum(axis=1).mean()
    d["GL"] = w.abs().sum(axis=1).mean()
    d["PL"] = (w < 0).mean(axis=1).mean()
    d["herf"] = (((w**2).sum(axis=1)) ** (-1)).mean()
    d['pos'] = w[w > 1e-7].T.count().mean()
    x = pd.Series(d)
    x.index.name = "Weight stats"
    return x


def build_table2(cov_models, ptf_ret):
    
    order_of_models = cov_models
    full_table = pd.DataFrame(index=order_of_models)
    
    for m in ptf_ret:
        perf = ptf_ret[m].apply(ptf_summary)
        perf = perf.loc[["AV", "SD", "IR", "VaR", "MDD", "P2T", "P2P", "P2PL",
                         "Calmar" , "Stability", "Omega", "Sortino", "TailRatio", "CSR", "Kurtosis" ]]
        perf = perf.applymap(lambda x: float(x))
        full_table.loc[m, "AV"]  = round(perf["Profit"]["AV"], 2)
        full_table.loc[m, "SD"]  = round(perf["Profit"]["SD"], 2)
        full_table.loc[m, "IR"]  = round(perf["Profit"]["IR"], 2)
        full_table.loc[m, "VaR"]  = round(perf["Profit"]["VaR"], 2)
        full_table.loc[m, "MDD"]  = round(perf["Profit"]["MDD"], 2)
        full_table.loc[m, "P2T"]  = round(perf["Profit"]["P2T"], 2)
        full_table.loc[m, "P2P"]  = round(perf["Profit"]["P2P"], 2)
        full_table.loc[m, "P2PL"]  = round(perf["Profit"]["P2PL"], 2)
        full_table.loc[m, "Calmar"]  = round(perf["Profit"]["Calmar"], 2)
        full_table.loc[m, "Stability"]  = round(perf["Profit"]["Stability"], 2)
        full_table.loc[m, "Omega"]  = round(perf["Profit"]["Omega"], 2)
        full_table.loc[m, "Sortino"]  = round(perf["Profit"]["Sortino"], 2)
        full_table.loc[m, "TailRatio"]  = round(perf["Profit"]["TailRatio"], 2)
        full_table.loc[m, "CSR"]  = round(perf["Profit"]["CSR"], 2)
        full_table.loc[m, 'Kurtosis']  = round(perf["Profit"]['Kurtosis'], 2)
    
    return full_table


def build_table3(cov_models, w, ptf_ret):
    
    order_of_models = cov_models
    full_table = pd.DataFrame(index=order_of_models)
    
    for m in ptf_ret:
        perf = weight_summary(w[m], ptf_ret[m])
        perf = perf.loc[["TO", "GL", "PL", "IR_net", "herf",  "pos"]]
#         perf = perf.applymap(lambda x: float(x))
        full_table.loc[m, "TO"]  = round(perf["TO"], 2)
        full_table.loc[m, "GL"]  = round(perf["GL"], 2)
        full_table.loc[m, "PL"]  = round(perf["PL"], 2)
        full_table.loc[m, "IR_net"]  = round(perf["IR_net"], 2)
        full_table.loc[m, 'herf']  = round(perf['herf'], 2)
        full_table.loc[m, 'pos']  = round(perf['pos'], 2)
    
    return full_table


def plotting(pnl_results, cov_models):
    color_legend = ['#4E87A0', '#008264', '#00A3AD', '#7C878E', '#9E2A2B', '#003865','#E53D51','#A2C663','#4698CB']

    pnl_fig = go.Figure(
        data=[
            go.Scatter(
                x=pnl_results[m].index,
                y=pnl_results[m]['NAV'],
                name=m,
                opacity=0.8,
                line=dict(color=color_legend[i], width=2)
            ) for i,m in enumerate(cov_models)
        ],
        layout = go.Layout(
            title='Cumulative PnL',
            yaxis=dict(
            title='Cumulative Returns (Log-Scale)',
            type='log'
        ),
        )
    )
    pnl_fig.show()

    
########################################################################################################################


import numpy as np
from scipy.stats import linregress
from itertools import accumulate
from scipy.stats import norm


def _convert_to_array(x):
    v = np.asanyarray( x )
    v = v[ np.isfinite(v) ]
    return v


def annual_return(returns, price_rets='price', trading_days=252):
    '''
    Computing the average compounded return (yearly)
    '''
    assert trading_days > 0, 'Trading days needs to be > 0'
    v = _convert_to_array(returns)
    n_years = v.size / float(trading_days)
    if price_rets == 'returns':
        return ( np.prod( (1. + v) ** (1. / n_years) ) - 1. if v.size > 0 else np.nan)
    else:
        return (np.sum(v)* (1. / n_years)  if v.size > 0 else np.nan)


def annual_volatility(returns, trading_days=252):
    assert trading_days > 0, 'Trading days needs to be > 0'
    v = _convert_to_array(returns)
    return (np.std(v) * np.sqrt(trading_days) if v.size > 0 else np.nan )


def value_at_risk(returns, horizon=10, pctile = 0.99, mean_adj=False):
    assert horizon>1, 'horizon>1'
    assert pctile<1
    assert pctile>0, 'pctile in [0,1]'
    v = _convert_to_array(returns)
    stdev_mult = norm.ppf(pctile) # i.e., 1.6449 for 0.95, 2.326 for 0.99
    if mean_adj:
        gains = annual_return(returns,  'price', horizon)
    else:
        gains = 0

    return (np.std(v) * np.sqrt(horizon) * stdev_mult - gains if v.size > 0 else np.nan)


def sharpe_ratio(returns, risk_free=0., trading_days=252):
    assert trading_days > 0, 'Trading days needs to be > 0'
    v = _convert_to_array(returns - risk_free)
    return (np.mean(v) / np.std(v) * np.sqrt(trading_days) if v.size > 0 else np.nan)


def max_drawdown(returns, price_rets ='price'):
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    if price_rets == 'returns':
        cumret = np.concatenate(([100.], 100. * np.cumprod(1. + v)))
        maxret = np.fmax.accumulate(cumret)
        return np.nanmin((cumret - maxret) / maxret)
    else:
        cumret = np.concatenate(([1.], np.cumsum(v)))
        maxret = np.fmax.accumulate( cumret )
        return np.nanmin(cumret - maxret)


def max_drawdown_length(returns, price_rets ='price'):
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    drawndown = np.zeros(len(v)+1)
    dd_dict = dict()
    if price_rets == 'returns':
        cumret = np.concatenate(([100.], 100. * np.cumprod(1. + v)))
        maxret = np.fmax.accumulate(cumret)
        drawndown[(cumret - maxret) / maxret < 0 ] = 1
    else:
        cumret = np.concatenate(([1.], np.cumsum(v))) # start at one? no matter
        maxret = np.fmax.accumulate(cumret)
        drawndown[(cumret - maxret) < 0] = 1

    f = np.frompyfunc((lambda x, y : (x + y) * y), 2, 1)
    run_lengths = f.accumulate(drawndown, dtype='object').astype(int)


    trough_position = np.argmin(cumret-maxret)
    peak_to_trough = run_lengths[trough_position]

    next_peak_rel_position = np.argmin(run_lengths[trough_position: ])
    next_peak_position = next_peak_rel_position + trough_position

    if run_lengths[next_peak_position] > 0:  # We are probably still in DD
        peak_to_peak = np.nan
    else:
        peak_to_peak = run_lengths[next_peak_position -1]
        # run_lengths just before it hits 0 (back to peak) is the
        # total run_length of that DD period.

    longest_dd_length = max(run_lengths) # longest, not nec deepest

    dd_dict['peak_to_trough_maxdd'] = peak_to_trough
    dd_dict['peak_to_peak_maxdd'] = peak_to_peak
    dd_dict['peak_to_peak_longest'] = longest_dd_length
    return dd_dict


def calmar_ratio(returns, price_rets = 'price', trading_days=252):
    assert trading_days > 0, 'Trading days needs to be > 0'
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    maxdd = max_drawdown(v, price_rets)
    if np.isnan( maxdd ):
        return np.nan
    annret = annual_return(v, price_rets, trading_days=trading_days)
    return annret / np.abs(maxdd)


def stability_of_timeseries(returns, price_rets = 'price'):
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    if price_rets == 'returns':
        v = np.cumsum( np.log1p( v ) )
    else:
        v = np.cumsum(v)
    lin_reg = linregress(np.arange(v.size), v)
    return lin_reg.rvalue**2


def omega_ratio(returns, risk_free=0., target_return=0., trading_days=252):
    assert trading_days > 0, 'Trading days needs to be > 0'
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    return_thresh = (1. + target_return) ** (1. / trading_days) - 1.
    v = v - risk_free - return_thresh
    numer = np.sum( v[v > 0.] )
    denom = -np.sum( v[v < 0.] )
    return (numer / denom if denom > 0. else np.nan)


def sortino_ratio(returns, target_return=0., trading_days=252):
    assert trading_days > 0, 'Trading days needs to be > 0'
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    v = v - target_return
    downside_risk = np.sqrt( np.mean( np.square( np.clip(v, np.NINF, 0.) )))
    return np.mean(v) * np.sqrt(trading_days) / downside_risk


def tail_ratio(returns):
    v = _convert_to_array(returns)
    return (np.abs(np.percentile(v, 95.)) / np.abs(np.percentile(v, 5.)) if v.size > 0 else np.nan)


def common_sense_ratio(returns):
    # This cannot be compared with pyfolio routines because they implemented a
    # wrong formula CSR = Tail Ratio * Gain-to-Pain Ratio
    # and Gain-to-Pain Raio = Sum(Positive R) / |Sum(Negative R)|
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    return tail_ratio(returns) * np.sum(v[v > 0.]) / np.abs(np.sum(v[v < 0.]))
