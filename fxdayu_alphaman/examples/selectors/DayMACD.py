# encoding:utf-8

import datetime
import pandas as pd
from fxdayu_data import DataAPI
from talib.abstract import MACD

from fxdayu_alphaman.selector.selector import Selector


class DayMACD(Selector):

    # 在此处设置选股器相关参数

    fastperiod = 12
    slowperiod = 26
    signalperiod = 9


    max_window = 30 #最大回溯单位时间数 可选 默认100



    def calculate_MACD_signal(self, data):
        candle_data = data[1].dropna()
        if len(candle_data) == 0:
            return
        hist = MACD(candle_data,
                    fastperiod= self.fastperiod,
                    slowperiod= self.slowperiod,
                    signalperiod= self.signalperiod)["macdhist"]

        hist_over_0 = hist >0
        hist_under_0 = hist <=0
        cross_over = (hist_under_0.shift(1)) * hist_over_0
        cross_under = - (hist_over_0.shift(1))*hist_under_0
        choice = pd.DataFrame(cross_over + cross_under)
        choice.columns = [data[0],]
        return choice


    def execute(self, pool, start, end, data=None, data_config=None):
        """
        计算选股结果

        :param pool: list 待选股票池
               data：进行选股操作所需要的数据
               start:datetime 选股索引范围起始时间
               end：datetime 选股索引范围结束时间
        :return:selector_result:A MultiIndex Series indexed by date (level 0) and asset (level 1), containing
        the values mean whether choose the asset or not.

            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   1
                        -----------------------
                        |   BA       |  1
                        -----------------------
            2014-01-01  |   CMG      |   1
                        -----------------------
                        |   DAL      |  0
                        -----------------------
                        |   LULU     |   -1
                        -----------------------
        """

        if not data:
            data = DataAPI.get(symbols=tuple(pool),
                               start=start-datetime.timedelta(days=self.max_window) ,
                               end = end,
                               **data_config)

        selector_result = map(self.calculate_MACD_signal, data.iteritems())
        selector_result = pd.concat(selector_result, axis=1).stack()
        selector_result.index.names = ["date","asset"]
        return selector_result.loc[start:end]


