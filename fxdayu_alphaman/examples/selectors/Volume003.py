# encoding:utf-8

import datetime

import alphalens
from fxdayu_data import DataAPI

from fxdayu_alphaman.examples.factors.Factor_Volume003 import Factor_Volume003
from fxdayu_alphaman.selector import Selector


class Volume003(Selector):

    # 在此处设置选股器相关参数
    quantiles = 5
    choose_quantile = 5 #选quantile最大的

    max_window = 30 #最大回溯单位时间数 可选 默认100

    def get_quantiles(self,factor):
        quantiles = alphalens.utils.quantize_factor(factor, quantiles=self.quantiles)
        return quantiles

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

        if data is None:
            data = DataAPI.get(symbols=tuple(pool),
                               start=start-datetime.timedelta(days=self.max_window) ,
                               end = end,
                               **data_config)

        factor = Factor_Volume003().get_factor(data , update= True)

        quantiles = self.get_quantiles(factor)
        selector_result = quantiles.copy()
        selector_result[:] = 0
        selector_result[quantiles == self.choose_quantile] = 1

        return selector_result.loc[start:end]


