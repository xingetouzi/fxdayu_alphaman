# encoding:utf-8

import datetime
import pandas as pd
from fxdayu_data import DataAPI
from talib.abstract import MA

from fxdayu_alphaman.selector.selector import Selector


class DayMA(Selector):

    # 在此处设置选股器相关参数
    fast = 5
    slow = 10

    max_window = 30 #最大回溯单位时间数 可选 默认100



    def calculate_MA_signal(self, data):
        candle_data = data[1].dropna()
        if len(candle_data) == 0:
            return
        fast = MA(candle_data, timeperiod=self.fast)
        slow = MA(candle_data, timeperiod=self.slow)
        f_over_s = fast - slow >0
        f_under_s = fast - slow <=0
        cross_over = (f_under_s.shift(1)) * f_over_s
        cross_under = - (f_over_s.shift(1))*f_under_s
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

        selector_result = map(self.calculate_MA_signal, data.iteritems())
        selector_result = pd.concat(selector_result, axis=1).stack()
        selector_result.index.names = ["date","asset"]
        return selector_result.loc[start:end]


