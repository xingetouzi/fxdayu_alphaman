# coding=utf-8

import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy.stats import rankdata
import talib as ta
from fxdayu_alphaman.factor.utility import read_LFLO,get_industry_class

# factor基类
class Factor(object):
    __name__ = 'factor'

    # 记录最大数据边界回溯视窗
    max_window = 100

    def __init__(self):
        self._factor = None

    #获得因子值
    def get_factor(self, pn_data, update=False):
        """
        获得因子值
        :param pn_data:  pandas.Panel类型 一共三个维度。items axis 为股票代码,包含所有该因子跟踪的股票;
                         在这之下,每只股票对应一个pandas.Dataframe,对应这只股票需要被访问到的数据,index为时间,columns为数据字段名称。
                         如:<class 'pandas.core.panel.Panel'>
                            Dimensions: 196 (items) x 335 (major_axis) x 5 (minor_axis)
                            Items axis: sh600011 to sz300450
                            Major_axis axis: 2015-12-02 15:00:00 to 2017-04-18 15:00:00
                            Minor_axis axis: close to volume
        :param update: 是否更新因子值(bool),默认为False,即在该factor实例被销毁前,因子值以第一次计算为准。否则每次取因子值都会重新计算。
        :return: factor_value: A MultiIndex Series indexed by date (level 0) and asset (level 1), containing
        the values for a single alpha factor.
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
        """
        if (self._factor is None) or update:
            self._factor = self.factor_calculator(pn_data)
            return self._factor
        else:
            return self._factor

    # 将因子用排序分值重构 默认为升序——因子越大 排序分值越大
    @staticmethod
    def get_factor_by_rankScore(factor, ascending = True):
        """
        输入multiIndex格式的因子值, 将因子用排序分值重构，并处理到0-1之间(默认为升序——因子越大 排序分值越大(越好)
        :param factor:  A MultiIndex Series indexed by date (level 0) and asset (level 1), containing
        the values for a single alpha factor.
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
        :param ascending: 因子值按升序法排序对应还是降序法排序对应。具体根据因子对收益的相关关系而定，为正则应用升序,为负用降序。(bool)
        :return: 重构后的因子值。A MultiIndex Series indexed by date (level 0) and asset (level 1), containing
        the values for a single alpha factor. 取值范围在0-1之间
        """
        gather_rank = []
        for date in factor.index.levels[0]:
            gather_rank.append(factor.loc[date:date].rank(method="min", ascending=ascending))
        gather_rank = pd.concat(gather_rank)

        # 将rank后的因子映射到(0,1】
        rank_range = len(gather_rank.index.levels[1])
        if rank_range>0:
            gather_rank = gather_rank/rank_range
        return gather_rank

    # 横截面去极值 - 对Dataframe数据
    def winsorize(self, factor_df):
        """
        对因子值做去极值操作
        :param factor_df: 因子值 (pandas.Dataframe类型),index为datetime, colunms为股票代码。
                          形如:
                                      　AAPL	　　　     BA	　　　CMG	　　   DAL	      LULU	　　
                            date
                            2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832
                            2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890
                            2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691
                            2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805
                            2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902
        :return:去极值后的因子值(pandas.Dataframe类型),index为datetime, colunms为股票代码。
        """

        def winsorize_series(se):
            q = se.quantile([0.025, 0.975])
            if isinstance(q, pd.Series) and len(q) == 2:
                se[se < q.iloc[0]] = q.iloc[0]
                se[se > q.iloc[1]] = q.iloc[1]
            return se

        def handle(rows):
            return winsorize_series(rows[1])

        result = pd.DataFrame(list(map(handle, factor_df.iterrows())), factor_df.index)

        return result


    @staticmethod
    # 横截面标准化 - 对Dataframe数据
    def standardize(factor_df):
        """
        对因子值做z-score标准化
        :param factor_df: 因子值 (pandas.Dataframe类型),index为datetime, colunms为股票代码。
                          形如:
                                      　AAPL	　　　     BA	　　　CMG	　　   DAL	      LULU	　　
                            date
                            2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832
                            2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890
                            2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691
                            2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805
                            2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902
        :return:z-score标准化后的因子值(pandas.Dataframe类型),index为datetime, colunms为股票代码。
        """


        def standardize_series(se):
            se_std = se.std()
            se_mean = se.mean()
            return (se - se_mean) / se_std

        def handle(rows):
            return standardize_series(rows[1])

        result = pd.DataFrame(list(map(handle, factor_df.iterrows())), factor_df.index)

        return result

    # 行业、市值中性化 - 对Dataframe数据
    def neutralize(self, factor_df, factorIsMV = False):
        """
        对因子做行业、市值中性化
        :param factor_df: 因子值 (pandas.Dataframe类型),index为datetime, colunms为股票代码。
                          形如:
                                      　AAPL	　　　     BA	　　　CMG	　　   DAL	      LULU	　　
                            date
                            2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832
                            2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890
                            2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691
                            2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805
                            2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902
        :param factorIsMV: 待中性化的因子是否是市值类因子(bool)。是则为True,默认为False
        :return: 中性化后的因子值(pandas.Dataframe类型),index为datetime, colunms为股票代码。
        """

        # 剔除有过多无效数据的个股
        empty_data = pd.isnull(factor_df).sum()
        pools = empty_data[empty_data < len(factor_df) * 0.1].index  # 保留空值比例低于0.1的股票
        factor_df = factor_df.loc[:, pools]

        # 剔除过多值为空的截面
        factor_df = factor_df.dropna(thresh = len(factor_df.columns) * 0.9) # 保留空值比例低于0.9的截面

        # 获取行业分类信息
        X = get_industry_class(pools)
        nfactors = X.index[-1]

        # 获取对数流动市值，并去极值、标准化。市值类因子不需进行这一步
        if not factorIsMV:
            x1 = self.standardize(self.winsorize(read_LFLO(pools, factor_df.index[0], factor_df.index[-1])))
            nfactors += 1

        result = []
        # 逐个截面进行回归，留残差作为中性化后的因子值
        for i in factor_df.index:
            if not factorIsMV:
                DataAll = pd.concat([X.T, x1.loc[i], factor_df.loc[i]], axis=1)
            else:
                DataAll = pd.concat([X.T, factor_df.loc[i]], axis=1)
            # 剔除截面中值含空的股票
            DataAll = DataAll.dropna()
            DataAll.columns = list(range(0, nfactors + 1))
            regr = linear_model.LinearRegression(fit_intercept=False)
            regr.fit(np.matrix(DataAll.iloc[:, 0:nfactors]), np.transpose(np.matrix(DataAll.iloc[:, nfactors])))
            residuals = np.transpose(np.matrix(DataAll.iloc[:, nfactors])) -regr.predict(np.matrix(DataAll.iloc[:, 0:nfactors]))
            residuals = pd.DataFrame(data=residuals, index=np.transpose(np.matrix(DataAll.index.values)))
            residuals.index = DataAll.index.values
            residuals.columns = [i]
            result.append(residuals)

        result = pd.concat(result, axis=1).T

        return result

    @staticmethod
    # 将pandas.Dataframe格式的因子值调整为MultiIndex格式
    def factor_df_to_factor_mi(factor_df):
        """
        将pandas.Dataframe格式的因子值调整为MultiIndex格式 如下
        :param factor_df: 因子值 (pandas.Dataframe类型),index为datetime, colunms为股票代码(asset)。
                          形如:
                                      　AAPL	　　　     BA	　　　CMG	　　   DAL	      LULU	　　
                            date
                            2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832
                            2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890
                            2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691
                            2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805
                            2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902
        :return:factor_value: A MultiIndex Series indexed by date (level 0) and asset (level 1), containing
        the values for a single alpha factor.
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
        """

        factor_data = factor_df.stack()
        factor_data = pd.DataFrame(factor_data)
        factor_data.reset_index(inplace=True)
        factor_data.columns = ["date", "asset", "factor"]
        factor_data.set_index(["date","asset"], inplace=True)
        return factor_data

    # 计算因子
    def factor_calculator(self, pn_data):
        """
        在此方法里实现因子的计算逻辑

        :param pn_data:  时间为索引，按资产分类的一张时间序列数据表，用于计算因子的初始所需数据
                         pandas.Panel类型 一共三个维度。items axis 为股票代码,包含所有该因子跟踪的股票;
                         在这之下,每只股票对应一个pandas.Dataframe,对应这只股票需要被访问到的数据,index为时间,columns为数据字段名称。
                         如:<class 'pandas.core.panel.Panel'>
                            Dimensions: 196 (items) x 335 (major_axis) x 5 (minor_axis)
                            Items axis: sh600011 to sz300450
                            Major_axis axis: 2015-12-02 15:00:00 to 2017-04-18 15:00:00
                            Minor_axis axis: close to volume

        :return:facor_value:A MultiIndex Series indexed by date (level 0) and asset (level 1), containing
        the values for a single alpha factor.

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
        """
        return pd.DataFrame()


    '''
    Basic_Factor_Functions
    '''

    # 移动求和
    def ts_sum(self, se, window=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.rolling(window).sum()

    #移动平均
    def ts_mean(self, se, window=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.rolling(window).mean()

    #标准差
    def stddev(self, se, window=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.rolling(window).std()

    #相关系数
    def correlation(self, se_x, se_y, window=10):
        if not isinstance(se_x,pd.Series):
            se_x = pd.Series(se_x)
        if not isinstance(se_y,pd.Series):
            se_y = pd.Series(se_y)
        return se_x.rolling(window).corr(se_y)

    #协方差
    def covariance(self, se_x, se_y, window=10):
        if not isinstance(se_x,pd.Series):
            se_x = pd.Series(se_x)
        if not isinstance(se_y,pd.Series):
            se_y = pd.Series(se_y)
        return se_x.rolling(window).cov(se_y)

    def rolling_rank(self, se):
        return rankdata(se)[-1]

    #移动排序
    def ts_rank(self, se, window=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.rolling(window).apply(self.rolling_rank)

    def rolling_prod(self, se):
        return se.prod(se)


    def product(self, se, window=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.rolling(window).apply(self.rolling_prod)

    # 移动窗口最小值
    def ts_min(self, se, window=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.rolling(window).min()

    # 最大值
    def ts_max(self, se, window=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.rolling(window).max()

    #delta
    def delta(self, se, period=1):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.diff(period)

    # shift
    def delay(self, se, period=1):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.shift(period)

    # 排序
    def rank(self, se):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.rank(axis=1, pct=True)

    def scale(self, se, k=1):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.mul(k).div(np.abs(se).sum())

    def ts_argmax(self, se, window=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.rolling(window).apply(np.argmax) + 1

    def ts_argmin(self, se, window=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return se.rolling(window).apply(np.argmin) + 1

    def decay_linear(self, se, period=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return ta.WMA(se.values, period)

    # 斜率
    def slope(self, se, period=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return ta.LINEARREG_SLOPE(se.values, period)

    def atr(self, df):
        if not isinstance(df,pd.DataFrame):
            df = pd.Series(df)
        return ta.abstract.ATR(df)

    def momentum(self, se, period=10):
        if not isinstance(se,pd.Series):
            se = pd.Series(se)
        return ta.ROCP(se.values, period)
