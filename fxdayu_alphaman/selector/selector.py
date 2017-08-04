# encoding:utf-8
import pandas as pd


class Selector(object):

    __name__ = 'selector'

    # 记录最大数据边界回溯视窗
    max_window = 100


    def __init__(self, data=None, data_config=None):
        ## optional 可在选股器初始化时指定选股所用的数据集（data)，或通过fxdayu_data api指定所用数据集；也可在selector_result 方法里单独指定。
        # 存储选股结果
        self._selector_result = None
        self.data = data
        self.data_config = data_config if data_config is not None else {"freq": "D", "api": "candle", "adjust":"after"}

    # 获得选股结果
    def selector_result(self, pool, start, end, data=None, data_config=None, update=False):
        """
        获得选股结果
        :param pool: 股票池范围（list),如：["000001.XSHE","600300.XSHG",......]
        :param start: 起始时间 (datetime)
        :param end: 结束时间 (datetime)
        :param data (optional): 计算选股结果需用到的数据,根据计算需求自行指定。(可选)
        :param data_config (optional): 在data参数为None的情况下(不传入自定义数据),
                                       可通过该参数调用dxdayu_data api 访问到数据 (dict),
                                       与data参数二选一。
        :param update: 是否更新计算(bool)。默认为False(当该选股器实例曾经做过选股运算时,再次调用该方法将不会重新计算,而直接返回上次计算的结果)。
        :return: selector_result: 选股器结果　格式为一个MultiIndex Series，索引(index)为date(level 0)和asset(level 1),
                            　　　 包含一列结果值。(1:选出,0:不选,-1:做空)  形如:
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

        if (self._selector_result is not None) and not update:
            return self._selector_result.loc[start:end]
        else:
            if data is None:
                data = self.data
            if data_config is None:
                data_config = self.data_config
            self._selector_result = self.execute(pool, start, end, data, data_config)
            return self._selector_result

    #执行选股计算
    def execute(self, pool, start, end, data=None, data_config=None):
        """
        执行选股计算
        :param pool: 股票池范围（list),如：["000001.XSHE","600300.XSHG",......]
        :param start: 起始时间 (datetime)
        :param end: 结束时间 (datetime)
        :param data (optional): 计算选股结果需用到的数据,根据计算需求自行指定。(可选)
        :param data_config (optional): 在data参数为None的情况下(不传入自定义数据),
                                       可通过该参数调用fxdayu_data api 访问到数据 (dict),
                                       与data参数二选一。
        :return: selector_result: 选股器结果　格式为一个MultiIndex Series，索引(index)为date(level 0)和asset(level 1),
                            　　　 包含一列结果值。(1:选出,0:不选,-1:做空)  形如:
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

        return pd.MultiIndex()

