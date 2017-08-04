# -*- coding: utf-8 -*-

from alphalens import utils,performance
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter


class NonMatchingTimezoneError(Exception):
    pass

def get_clean_factor_and_forward_returns(factor,
                                         prices,
                                         groupby=None,
                                         by_group=False,
                                         quantiles=5,
                                         bins=None,
                                         periods=(1, 5, 10),
                                         filter_zscore=20,
                                         groupby_labels=None):
    """
    Formats the factor data, pricing data, and group mappings
    into a DataFrame that contains aligned MultiIndex
    indices of date and asset.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1), containing
        the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets
        in the columns. It is important to pass the
        correct pricing data in depending on what time of period your
        signal was generated so to avoid lookahead bias, or
        delayed calculations. Pricing data must span the factor
        analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    by_group : bool
        If True, compute statistics separately for each group.
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Only one of 'quantiles' or 'bins' can be not-None
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for each period,
        The factor quantile/bin that factor value belongs too, and (optionally) the group the
        asset belongs to.
    """

    if factor.index.levels[0].tz != prices.index.tz:
        raise NonMatchingTimezoneError("The timezone of 'factor' is not the "
                                       "same as the timezone of 'prices'. See "
                                       "the pandas methods tz_localize and tz_convert.")

    merged_data = utils.compute_forward_returns(prices, periods, filter_zscore)

    factor = factor.copy()
    factor.index = factor.index.rename(['date', 'asset'])
    merged_data['factor'] = factor

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(factor.index.get_level_values(
                'asset')) - set(groupby.keys())
            if len(diff) > 0:
                raise KeyError(
                    "Assets {} not in group mapping".format(
                        list(diff)))

            ss = pd.Series(groupby)
            groupby = pd.Series(index=factor.index,
                                data=ss[factor.index.get_level_values(
                                    'asset')].values)

        if groupby_labels is not None:
            diff = set(groupby.values) - set(groupby_labels.keys())
            if len(diff) > 0:
                raise KeyError(
                    "groups {} not in passed group names".format(
                        list(diff)))

            sn = pd.Series(groupby_labels)
            groupby = pd.Series(index=factor.index,
                                data=sn[groupby.values].values)

        merged_data['group'] = groupby.astype('category')

    merged_data = merged_data.dropna()

    try:
        merged_data['factor_quantile'] = utils.quantize_factor(merged_data,
                                                               quantiles,
                                                               bins,
                                                               by_group)
    except:
        merged_data['factor_quantile'] = 1

    merged_data = merged_data.dropna()

    return merged_data

#获取一个时间序列列表，描述按某种持股方案得到的股票持有期收益
def get_stocks_holding_return(stock_strategy, prices, strategy_name="", periods=(1,5,10)):
    """
    计算持股方案的股票持有期(periods)收益
    :param stock_strategy: 一个MultiIndex Series的序列。索引为date (level 0) 和 asset (level 1),包含一列持有股票相对权重(目前在计算总体收益时并未生效)。
                       其中asset的值为指定日期待持有的股票代码。形如:
                        -----------------------------------
                            date    |    asset   |
                        -----------------------------------
                                    |   AAPL     |   1
                                    -----------------------
                                    |   BA       |  1
                                    -----------------------
                        2014-01-01  |   CMG      |   1
                                    -----------------------
                                    |   DAL      |  3
                                    -----------------------
                                    |   LULU     |   2
                                    -----------------------

    :param prices: 计算绩效时用到的的个股每日价格,通常为收盘价（close）。
                   索引（index)为datetime,columns为各股票代码。pandas dataframe类型,形如:
                                       sh600011  sh600015  sh600018  sh600021  sh600028
                datetime
                2014-10-08 15:00:00    18.743    17.639       NaN     7.463     9.872
                2014-10-09 15:00:00    18.834    17.556       NaN     7.536     9.909
                2014-10-10 15:00:00    18.410    17.597       NaN     7.580     9.835
                2014-10-13 15:00:00    18.047    17.515       NaN     7.536     9.685
                2014-10-14 15:00:00    18.773    17.494       NaN     7.433     9.704
                2014-10-15 15:00:00    18.561    17.597       NaN     7.477     9.704
                2014-10-16 15:00:00    18.501    17.659       NaN     7.448     9.685
                2014-10-17 15:00:00    18.349    17.535       NaN     7.272     9.611
                2014-10-20 15:00:00    18.319    17.618       NaN     7.360     9.629
                .....................................................................
    :param strategy_name: 该持股方案的名称,可任意命名(str)
    :param periods: 持有周期(tuple)
    :return: 该持股方案对应每一只股票的持有期收益。(pd.Dataframe)。
    """
    stocks_return = get_clean_factor_and_forward_returns(stock_strategy, prices, quantiles=1, periods=periods)
    stocks_return = stocks_return.reset_index()
    stocks_return["factor_quantile"] = strategy_name

    return(stocks_return)

def compute_downside_returns(prices, prices_low, periods=(1, 5, 10), filter_zscore=None):
    """
    Finds the N period downside_returns (as percent change) for each asset provided.

    Parameters
    ----------
    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    prices_low : pd.DataFrame
        Low pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.

    Returns
    -------
    downside_returns : pd.DataFrame - MultiIndex
        downside_returns in indexed by date and asset.
        Separate column for each forward return window.
    """

    downside_returns = pd.DataFrame(index=pd.MultiIndex.from_product(
        [prices.index, prices.columns], names=['date', 'asset']))

    for period in periods:
        delta = (prices_low.rolling(period).min().shift(-period) - prices) / prices

        if filter_zscore is not None:
            mask = abs(delta - delta.mean()) > (filter_zscore * delta.std())
            delta[mask] = np.nan

            downside_returns[period] = delta.stack()

            downside_returns.index = downside_returns.index.rename(['date', 'asset'])

    return downside_returns


def get_downside_returns(factor,
                         prices,
                         prices_low,
                         periods=(1, 5, 10),
                         filter_zscore=20):
    """
    Finds the N period downside_returns (as percent change) for each asset provided.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1), containing
        the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------
    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    prices_low : pd.DataFrame
        Low pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.

    Returns
    -------
    downside_returns : pd.DataFrame - MultiIndex
        downside_returns in indexed by date and asset.
        Separate column for each forward return window and
        the values for a single alpha factor.
    """
    if factor.index.levels[0].tz != prices.index.tz:
        raise NonMatchingTimezoneError("The timezone of 'factor' is not the "
                                       "same as the timezone of 'prices'. See "
                                       "the pandas methods tz_localize and tz_convert.")

    merged_data = compute_downside_returns(prices, prices_low, periods, filter_zscore)

    factor = factor.copy()
    factor.index = factor.index.rename(['date', 'asset'])
    merged_data['factor'] = factor

    merged_data = merged_data.dropna()

    return merged_data

def get_stocks_downside_return(stock_strategy, prices, prices_low, strategy_name="", periods=(1,5,10)):
    """
    计算持股方案的股票下行(periods)收益
    :param stock_strategy: 一个MultiIndex Series的序列。索引为date (level 0) 和 asset (level 1),包含一列持有股票相对权重(目前在计算总体收益时并未生效)。
                       其中asset的值为指定日期待持有的股票代码。形如:
                        -----------------------------------
                            date    |    asset   |
                        -----------------------------------
                                    |   AAPL     |   1
                                    -----------------------
                                    |   BA       |  1
                                    -----------------------
                        2014-01-01  |   CMG      |   1
                                    -----------------------
                                    |   DAL      |  3
                                    -----------------------
                                    |   LULU     |   2
                                    -----------------------

    :param prices: 计算绩效时用到的的个股每日价格,通常为收盘价（close）。
                   索引（index)为datetime,columns为各股票代码。pandas dataframe类型,形如:
                                       sh600011  sh600015  sh600018  sh600021  sh600028
                datetime
                2014-10-08 15:00:00    18.743    17.639       NaN     7.463     9.872
                2014-10-09 15:00:00    18.834    17.556       NaN     7.536     9.909
                2014-10-10 15:00:00    18.410    17.597       NaN     7.580     9.835
                2014-10-13 15:00:00    18.047    17.515       NaN     7.536     9.685
                2014-10-14 15:00:00    18.773    17.494       NaN     7.433     9.704
                2014-10-15 15:00:00    18.561    17.597       NaN     7.477     9.704
                2014-10-16 15:00:00    18.501    17.659       NaN     7.448     9.685
                2014-10-17 15:00:00    18.349    17.535       NaN     7.272     9.611
                2014-10-20 15:00:00    18.319    17.618       NaN     7.360     9.629
                .....................................................................
    :param prices_low: 计算绩效时用到的的个股每日最低价格,pandas dataframe类型,形同prices
    :param strategy_name: 该持股方案的名称,可任意命名(str)
    :param periods: 持有周期(tuple)
    :return: 该持股方案对应每一只股票的下行收益。(pd.Dataframe)。
    """

    stocks_downside_return = get_downside_returns(stock_strategy, prices, prices_low, periods=periods)
    stocks_downside_return.fillna(0, inplace=True)
    stocks_downside_return= stocks_downside_return.reset_index()
    stocks_downside_return["factor_quantile"]=strategy_name

    return(stocks_downside_return)

def compute_upside_returns(prices, prices_high, periods=(1, 5, 10), filter_zscore=None):
    """
    Finds the N period upside_returns (as percent change) for each asset provided.

    Parameters
    ----------
    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    prices_high : pd.DataFrame like prices.
        High pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.

    Returns
    -------
    upside_returns : pd.DataFrame - MultiIndex
        upside_returns in indexed by date and asset.
        Separate column for each forward return window.
    """

    upside_returns = pd.DataFrame(index=pd.MultiIndex.from_product(
        [prices.index, prices.columns], names=['date', 'asset']))

    for period in periods:
        delta = (prices_high.rolling(period).max().shift(-period) - prices) / prices

        if filter_zscore is not None:
            mask = abs(delta - delta.mean()) > (filter_zscore * delta.std())
            delta[mask] = np.nan

            upside_returns[period] = delta.stack()

        upside_returns.index = upside_returns.index.rename(['date', 'asset'])

    return upside_returns

def get_upside_returns( factor,
                        prices,
                        prices_high,
                        periods=(1, 5, 10),
                        filter_zscore=20):
    """
    Finds the N period upside_returns (as percent change) for each asset provided.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1), containing
        the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------
    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    prices_high : pd.DataFrame
        High pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.

    Returns
    -------
    upside_returns : pd.DataFrame - MultiIndex
        upside_returns in indexed by date and asset.
        Separate column for each forward return window and
        the values for a single alpha factor.
    """

    if factor.index.levels[0].tz != prices.index.tz:
        raise NonMatchingTimezoneError("The timezone of 'factor' is not the "
                                       "same as the timezone of 'prices'. See "
                                       "the pandas methods tz_localize and tz_convert.")

    merged_data = compute_upside_returns(prices, prices_high, periods, filter_zscore)

    factor = factor.copy()
    factor.index = factor.index.rename(['date', 'asset'])
    merged_data['factor'] = factor

    merged_data = merged_data.dropna()

    return merged_data

def get_stocks_upside_return(stock_strategy, prices, prices_high, strategy_name="", periods=(1,5,10)):
    """
    计算持股方案的股票上行(periods)收益
    :param stock_strategy: 一个MultiIndex Series的序列。索引为date (level 0) 和 asset (level 1),包含一列持有股票相对权重(目前在计算总体收益时并未生效)。
                       其中asset的值为指定日期待持有的股票代码。形如:
                        -----------------------------------
                            date    |    asset   |
                        -----------------------------------
                                    |   AAPL     |   1
                                    -----------------------
                                    |   BA       |  1
                                    -----------------------
                        2014-01-01  |   CMG      |   1
                                    -----------------------
                                    |   DAL      |  3
                                    -----------------------
                                    |   LULU     |   2
                                    -----------------------

    :param prices: 计算绩效时用到的的个股每日价格,通常为收盘价（close）。
                   索引（index)为datetime,columns为各股票代码。pandas dataframe类型,形如:
                                       sh600011  sh600015  sh600018  sh600021  sh600028
                datetime
                2014-10-08 15:00:00    18.743    17.639       NaN     7.463     9.872
                2014-10-09 15:00:00    18.834    17.556       NaN     7.536     9.909
                2014-10-10 15:00:00    18.410    17.597       NaN     7.580     9.835
                2014-10-13 15:00:00    18.047    17.515       NaN     7.536     9.685
                2014-10-14 15:00:00    18.773    17.494       NaN     7.433     9.704
                2014-10-15 15:00:00    18.561    17.597       NaN     7.477     9.704
                2014-10-16 15:00:00    18.501    17.659       NaN     7.448     9.685
                2014-10-17 15:00:00    18.349    17.535       NaN     7.272     9.611
                2014-10-20 15:00:00    18.319    17.618       NaN     7.360     9.629
                .....................................................................
    :param prices_high: 计算绩效时用到的的个股每日最高价格,pandas dataframe类型,形同prices
    :param strategy_name: 该持股方案的名称,可任意命名(str)
    :param periods: 持有周期(tuple)
    :return: 该持股方案对应每一只股票的上行收益。(pd.Dataframe)。
    """

    stocks_upside_return = get_upside_returns(stock_strategy, prices, prices_high, periods=periods)
    stocks_upside_return.fillna(0, inplace=True)
    stocks_upside_return = stocks_upside_return.reset_index()
    stocks_upside_return["factor_quantile"] = strategy_name

    return (stocks_upside_return)

# 创建一个基准时间序列,将收益序列与之对齐
def align_return_series(return_series,start,end):
    """
    创建一个基准时间序列,将收益序列与之对齐
    :param return_series: 收益时序数据(pandas.multiIndex类型)。index 为factor_quantile——用以表示该收益序列的名称/标记 (level 0 )
                          和 date (level 1),colunms为不同持有期。
                          形如:
                                                                     1         5         10
                            factor_quantile date
                            hs300           2016-01-04 15:00:00  0.002412 -0.080094 -0.097879
                                            2016-01-05 15:00:00  0.017544 -0.075621 -0.073491
                                            2016-01-06 15:00:00 -0.069334 -0.108461 -0.103234
                                            2016-01-07 15:00:00  0.020392 -0.022101 -0.064665
                                            2016-01-08 15:00:00 -0.050307 -0.072237 -0.073805
                                            2016-01-11 15:00:00  0.007286 -0.019333 -0.019909
                                            2016-01-12 15:00:00 -0.018606  0.002304 -0.085580
                                            2016-01-13 15:00:00  0.020815  0.005862 -0.071463
                                            2016-01-14 15:00:00 -0.031922 -0.043525 -0.114171
                                            2016-01-15 15:00:00  0.003848 -0.001690 -0.055356
                                            2016-01-18 15:00:00  0.029511 -0.000588 -0.073363
                                            2016-01-19 15:00:00 -0.015122 -0.087682 -0.081223
                                            2016-01-20 15:00:00 -0.029307 -0.076875 -0.071113
                                            2016-01-21 15:00:00  0.010421 -0.073860 -0.031347
                                            2016-01-22 15:00:00  0.004956 -0.053757 -0.048072
                                            .................................................

    :param start: 标定的起始时间(datetime)
    :param end: 标定的结束时间(datetime)
    :return:被基准时间序列标齐后的收益序列。空缺值用0填充。确保收益序列在日期上连续
    """

    def time15(time):
        if time.hour == 15:
            return time
        else:
            return time.replace(hour=15)

    return_series = return_series.reset_index()

    # 将时间（date）全部标准化到收盘后（15:00:00）
    return_series["date"] = map(time15, return_series["date"])

    series_name = return_series["factor_quantile"].loc[0]

    timeS = pd.DataFrame(pd.date_range(start.strftime("%Y-%m-%d") + ' 15:00:00', periods=(end - start).days, freq='D'))
    timeS.columns =["date"]
    return_series =pd.merge(timeS,return_series,how="left")
    return_series["factor_quantile"].fillna(series_name,inplace=True)

    return(return_series.fillna(0))


def get_stocklist_mean_return(stock_strategy,strategy_name,start,end,prices,periods=(1,5,10)):
    """
    计算某持股方案的平均日持有收益
     :param stock_strategy: 一个MultiIndex Series的序列。索引为date (level 0) 和 asset (level 1),包含一列持有股票相对权重(目前在计算总体收益时并未生效)。
                       其中asset的值为指定日期待持有的股票代码。形如:
                        -----------------------------------
                            date    |    asset   |
                        -----------------------------------
                                    |   AAPL     |   1
                                    -----------------------
                                    |   BA       |  1
                                    -----------------------
                        2014-01-01  |   CMG      |   1
                                    -----------------------
                                    |   DAL      |  3
                                    -----------------------
                                    |   LULU     |   2
                                    -----------------------
    :param strategy_name: 该持股方案的名称,可任意命名(str)
    :param start: 起始时间(datetime)
    :param end: 终止时间(datetime)
    :param prices: 计算绩效时用到的的个股每日价格,通常为收盘价（close）。
                   索引（index)为datetime,columns为各股票代码。pandas dataframe类型,形如:
                                       sh600011  sh600015  sh600018  sh600021  sh600028
                datetime
                2014-10-08 15:00:00    18.743    17.639       NaN     7.463     9.872
                2014-10-09 15:00:00    18.834    17.556       NaN     7.536     9.909
                2014-10-10 15:00:00    18.410    17.597       NaN     7.580     9.835
                2014-10-13 15:00:00    18.047    17.515       NaN     7.536     9.685
                2014-10-14 15:00:00    18.773    17.494       NaN     7.433     9.704
                2014-10-15 15:00:00    18.561    17.597       NaN     7.477     9.704
                2014-10-16 15:00:00    18.501    17.659       NaN     7.448     9.685
                2014-10-17 15:00:00    18.349    17.535       NaN     7.272     9.611
                2014-10-20 15:00:00    18.319    17.618       NaN     7.360     9.629
                .....................................................................
    :param periods: 持有周期(tuple)
    :return: return_series: 日平均收益的时序数据(pandas.multiIndex类型)。index 为factor_quantile——用以表示该收益序列的名称/标记 (level 0 )
                          和 date (level 1),colunms为不同持有期。
                          形如:
                                                                     1         5         10
                            factor_quantile date
                            hs300           2016-01-04 15:00:00  0.002412 -0.080094 -0.097879
                                            2016-01-05 15:00:00  0.017544 -0.075621 -0.073491
                                            2016-01-06 15:00:00 -0.069334 -0.108461 -0.103234
                                            2016-01-07 15:00:00  0.020392 -0.022101 -0.064665
                                            2016-01-08 15:00:00 -0.050307 -0.072237 -0.073805
                                            2016-01-11 15:00:00  0.007286 -0.019333 -0.019909
                                            2016-01-12 15:00:00 -0.018606  0.002304 -0.085580
                                            2016-01-13 15:00:00  0.020815  0.005862 -0.071463
                                            2016-01-14 15:00:00 -0.031922 -0.043525 -0.114171
                                            2016-01-15 15:00:00  0.003848 -0.001690 -0.055356
                                            2016-01-18 15:00:00  0.029511 -0.000588 -0.073363
                                            2016-01-19 15:00:00 -0.015122 -0.087682 -0.081223
                                            2016-01-20 15:00:00 -0.029307 -0.076875 -0.071113
                                            2016-01-21 15:00:00  0.010421 -0.073860 -0.031347
                                            2016-01-22 15:00:00  0.004956 -0.053757 -0.048072
                                            .................................................
    """

    stocks_return = get_stocks_holding_return(stock_strategy,prices,strategy_name=strategy_name,periods=periods)

    # 获得持股平均收益
    mean_return = performance.mean_return_by_quantile(stocks_return.set_index(["date","asset"]), by_date=True, demeaned=False)[0]
    # 创建一个基准时间序列,将收益曲线与之对齐
    mean_return = align_return_series(mean_return,start,end)

    return(mean_return.set_index(["factor_quantile","date"]))


# 收益分布特征
def plot_distribution_features_table(stocks_return,periods):
    """
    根据指定持有周期下的收益序列(含持有收益、上行收益、下行收益等),计算该收益序列的分布特征。
    :param stocks_return: pandas.Dataframe. columns包含"factor_quantile"——收益序列名称/标记和periods当中存在的持有期。
                          如:
                           date        asset         1         5        10       factor_quantile
            0   2016-01-04 15:00:00  600600.XSHG  0.002412 -0.080094 -0.097879       test
            1   2016-01-04 15:00:00  601600.XSHG  0.017544 -0.075621 -0.073491       test
            2   2016-01-04 15:00:00  603600.XSHG -0.069334 -0.108461 -0.103234       test
            3   2016-01-04 15:00:00  600131.XSHG  0.020392 -0.022101 -0.064665       test
            4   2016-01-05 15:00:00  600600.XSHG -0.050307 -0.072237 -0.073805       test
            5   2016-01-15 15:00:00  000030.XSHE  0.007286 -0.019333 -0.019909       test
            6   2016-01-15 15:00:00  000031.XSHE -0.018606  0.002304 -0.085580       test
            7   2016-01-15 15:00:00  000032.XSHE  0.020815  0.005862 -0.071463       test
            8   2016-01-15 15:00:00  000034.XSHE -0.031922 -0.043525 -0.114171       test
            9   2016-01-15 15:00:00  000035.XSHE  0.003848 -0.001690 -0.055356       test
            10  2016-01-18 15:00:00  600600.XSHG  0.029511 -0.000588 -0.073363       test
            11  2016-01-18 15:00:00  601600.XSHG -0.015122 -0.087682 -0.081223       test
            12  2016-01-20 15:00:00  000030.XSHE -0.029307 -0.076875 -0.071113       test
            13  2016-01-20 15:00:00  000031.XSHE  0.010421 -0.073860 -0.031347       test
            14  2016-01-20 15:00:00  000032.XSHE  0.004956 -0.053757 -0.048072       test
            15  2016-01-20 15:00:00  000034.XSHE -0.060207 -0.072818 -0.058225       test
            16  2016-01-20 15:00:00  000035.XSHE -0.003455  0.007080  0.032828       test
    :param periods:持有时间(tuple)
    :return: 不同持有周期下的收益分布特征(pandas.Dataframe).形如:
                                    std      q0.5      q0.25      min       max      mean
holding period factor_quantile
1              DayMA              0.026118 -0.000401 -0.011203 -0.100128   0.100398  0.000648
5              DayMA              0.060400  0.001621 -0.024587 -0.285132   0.480710  0.004665
10             DayMA              0.075704  0.008019 -0.032575 -0.358760   0.673005  0.010801


    """
    frame=[]
    group = stocks_return.groupby('factor_quantile')
    for period in periods:
        distribution_features =group[period].\
            agg({"max": 'max',"min": 'min', "mean": "mean", "std": "std", 'q0.5': lambda x: x.quantile(0.5), 'q0.25': lambda x: x.quantile(0.25)})
        distribution_features["holding period"] = period
        frame.append(distribution_features)

    frame=pd.concat(frame)
    frame = frame.reset_index()
    frame.set_index(["holding period", "factor_quantile"], inplace=True)

    return(frame)

def plot_cumulative_returns(stocklist_mean_return, period=1, ax=None):
    """
    Plots the cumulative returns of various factor quantiles.

    Parameters
    ----------
    stocklist_mean_return : pd.MultiIndex
    period: int, optional
        Period over which the daily returns are calculated
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))


    ret_wide = stocklist_mean_return.reset_index()\
        .pivot(index='date', columns='factor_quantile', values=period)

    if period > 1:
        compound_returns = lambda ret, period: ( (np.nanmean(ret) + 1)**(1./period) ) - 1
        ret_wide = pd.rolling_apply(ret_wide, period, compound_returns,
                                    min_periods=1, args=(period,))

    cum_ret = ret_wide.add(1).cumprod()
    cum_ret = cum_ret.loc[:, ::-1]

    cum_ret.plot(lw=2, ax=ax, cmap=cm.RdYlGn_r)
    ax.legend()
    ymin, ymax = cum_ret.min().min(), cum_ret.max().max()
    ax.set(ylabel='Log Cumulative Returns',
           title='Cumulative Return ({} Period Forward Return)'.format(
               period),
           xlabel='',
           yscale='symlog',
           yticks=np.linspace(ymin, ymax, 5),
           ylim=(ymin, ymax))

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax

def plot_distribution_of_returns(returns, period, return_type="", ax=None):
    """
    Plots distribution_of_returns

    Parameters
    ----------
    returns : pd.DataFrame
        period holding returns.
    period: int
        Period over which the daily returns are calculated
    return_type: str
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    ax.legend()
    ax.set(ylabel='Frequency',
           title=' Distribution of %s Return (Holding Period %s)'%(return_type, period))
    data=returns[period]
    sns.distplot(data.multiply(100),axlabel=str(returns["factor_quantile"].loc[0]))

    return ax

def plot_stock_returns_violin(returns,
                              return_type="",
                              ax=None):
    """
    Plots a violin box plot of period wise returns for stocks.

    Parameters
    ----------
    returns : pd.DataFrame
        period holding returns.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    returns = returns.copy()

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    unstacked_dr = returns.drop(["date","asset","factor"],axis=1)
    unstacked_dr = unstacked_dr.set_index(["factor_quantile"])
    unstacked_dr = unstacked_dr.multiply(100)
    unstacked_dr.columns = unstacked_dr.columns.set_names('forward_periods')
    unstacked_dr = unstacked_dr.stack()
    unstacked_dr.name = 'return'
    unstacked_dr = unstacked_dr.reset_index()


    sns.violinplot(data=unstacked_dr,
                   x='factor_quantile',
                   hue='forward_periods',
                   y='return',
                   orient='v',
                   cut=0,
                   inner='quartile',
                   ax=ax)
    ax.set(xlabel='', ylabel='Return',
           title="Violin Box of Stocks'%s Period Return"%(return_type,))

    ax.axhline(0.0, linestyle='-', color='black', lw=0.7, alpha=0.6)

    return ax