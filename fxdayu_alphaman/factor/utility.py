# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from fxdayu_data import DataAPI

class MultiFactor(object):
    __slots__ = ["name", "multifactor_value", "weight"]

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, item):
        return self.__getattribute__(item)

class Benchmark(object):
    __slots__ = ["index", "close", "open", "high", "low"]

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, item):
        return self.__getattribute__(item)

class Performance(object):
    __slots__ = ["factor_name", "holding_return", "mean_return_by_q", "ic", "mean_ic_by_M", "mean_ic"]

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, item):
        return self.__getattribute__(item)


#获取指数行情
def read_benchmark(start, end, index_code="000300.XSHG", freq="D"):
    """
    获取指数行情数据
    :param start:起始时间(datetime)
    :param end:结束时间(datetime)
    :param index_code:指数代码(str)
    :param freq:数据频率(str)
    :return:指数行情。返回一个benchmark对象
                     包含五个属性。"index", "close", "open", "high", "low"。
                     "index":一个MultiIndex Series的序列。索引为date (level 0) 和 asset (level 1),包含一列factor值。其中asset的值固定是index_code,
                      factor值固定是1。用于fxdayu_alphaman.selector.selector_analysis.get_stocklist_mean_return中
                      作为stock_list参数来计算指数收益。
                     "close":收盘价。(pandas.Dateframe ),index为datetime,column.name 为index_code,值为对应指数的收盘价。
                     "open":开盘价。(pandas.Dateframe ),index为datetime,column.name 为index_code,值为对应指数的开盘价。
                     "high":最高价。(pandas.Dateframe ),index为datetime,column.name 为index_code,值为对应指数的最高价。
                     "low":最低价。(pandas.Dateframe ),index为datetime,column.name 为index_code,值为对应指数的最低价。
    """
    benchmark = Benchmark()
    benchmark_value = DataAPI.candle((index_code,), freq=freq, start=start, end=end)
    benchmark.open = benchmark_value.minor_xs("open")
    if len(benchmark.open)==0:
        raise ValueError
    benchmark.high = benchmark_value.minor_xs("high")
    benchmark.low = benchmark_value.minor_xs("low")
    benchmark.close = benchmark_value.minor_xs("close")
    benchmark.open.index.name = "date"
    benchmark.high.index.name = "date"
    benchmark.low.index.name = "date"
    benchmark.close.index.name = "date"

    benchmark.index = pd.DataFrame(data=benchmark.open.index)
    benchmark.index["asset"] = index_code
    benchmark.index["factor"] = 1
    benchmark.index = benchmark.index.set_index(["date", "asset"])
    return benchmark

def standard_code_style(symbols):
    """
    代码编码方式转化(sina to standard)
    :param symbols: 按sina标准制定的一组股票代码(list),形式为交易所+编码 如["sz000001","sh600000"]
    :return: 通用标准制定的一组股票代码(list),形式为编码.交易所 如["000001.XSHE","600000.XSHG"]
    """
    if len(symbols) == 0:
        return symbols
    if symbols[0].find("XSHE") >= 0 or symbols[0].find("XSHG") >= 0:
        return symbols

    new_symbols = []
    for symbol in symbols:
        if symbol.find("sz") == 0:
            new_symbol = symbol[2:8]+".XSHE"
        else:
            new_symbol = symbol[2:8]+".XSHG"
        new_symbols.append(new_symbol)
    return new_symbols

def sina_code_style(symbols):
    """
    代码编码方式转化(standard to sina)
    :param symbols: 通用标准制定的一组股票代码(list),形式为编码.交易所 如["000001.XSHE","600000.XSHG"]
    :return: 按sina标准制定的一组股票代码(list),形式为交易所+编码 如["sz000001","sh600000"]
    """
    if len(symbols) == 0:
        return symbols
    if symbols[0].find("sh") == 0 or symbols[0].find("sz") == 0:
        return symbols

    new_symbols = []
    for symbol in symbols:
        if symbol.find("SHE") >=0:
            new_symbol = "sz"+symbol[0:6]
        else:
            new_symbol = "sh"+symbol[0:6]
        new_symbols.append(new_symbol)
    return new_symbols

# 获取行业分类
def get_industry_class(symbols):
    """
    获取行业分类信息
    :param symbols: 一组股票代码(list),形式为通用标准(编码.交易所 如["000001.XSHE","600000.XSHG"])
    :return: sina的行业分类信息。(pandas.Dataframe) index为行业分类编号(1-49);columns为股票代码;值为0/1,分别表示属于该行业/不属于该行业
    """

    sina_industy_class = DataAPI.info.classification()
    sina_industy_class["classification"] = sina_industy_class["classification"].rank(method="dense", ascending=True).astype(int)
    class_num = sina_industy_class["classification"].max()
    frame = pd.DataFrame(0,index = np.arange(class_num)+1, columns = symbols)
    for symbol in symbols:
        this_class = sina_industy_class[sina_industy_class["code"] == symbol]["classification"]
        frame.loc[this_class, symbol] = 1
    return frame

#获取对数流通市值
def read_LFLO(symbols, start, end):
    """
    获取对数流通市值
    :param symbols: 一组股票代码(list),形式为通用标准(编码.交易所 如["000001.XSHE","600000.XSHG"])
    :param start: 开始时间(datetime)
    :param end:结束时间(datetime)
    :return: 对数流通市值。(pandas.Dataframe) index为datetime,columns为股票代码,值为对应股票对应时间的对数流通市值。
    """
    LFLO = DataAPI.factor(tuple(symbols),fields="LFLO",start = start, end = end).minor_xs("LFLO")
    return LFLO
