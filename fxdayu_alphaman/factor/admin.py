# encoding:utf-8

from ipyparallel import Client
from functools import partial
import pandas as pd
import os


def _apply(func, *args, **kwargs):
    return partial(func, *args, **kwargs)()


# 通过因子名称获取指定因子
def _get_factor(factor_name, factor_package_name):
    import importlib
    module = importlib.import_module("%s.%s" % (factor_package_name,
                                                factor_name))
    factor = getattr(module,
                     factor_name)
    return factor


class Admin(object):
    PACKAGE_NAME = os.environ.get("FXDAYU_FACTOR_PACKAGE_NAME", "factors")

    def __init__(self, *all_factors_name):
        self._all_factors_name = all_factors_name
        self._all_factors_value = {}

    # 等权重加权合成因子
    def equal_weighted_factor(self,
                              factor_name_list,
                              factor_value_list):
        """
        将若干个因子等权重合成新因子
        :param factor_name_list: 若干因子名称组成的列表(list)
        　　　　                   如：['factor_name_1','factor_name_2',...]
        :param factor_value_list: 若干因子值组成的列表(list)，与factor_name_list一一对应。
                           　　　　 每个因子值格式为一个MultiIndex Series，索引(index)为date(level 0)和asset(level 1),
                            　　　　包含一列factor值。
        :return: MultiFactor　对象。包含三个属性：
        　　　　　"name"：合成的因子名称（str)
                "multifactor_value":合成因子值（MultiIndex Series,索引(index)为date(level 0)和asset(level 1),
                                    包含一列factor值)
                "weight": 加权方式 (str)
        """

        from utility import MultiFactor

        def strategy_fun(gather, unit):
            return gather + unit

        # 因子累加
        gather_result = reduce(strategy_fun, factor_value_list)
        multifactor_name = "+".join(factor_name_list)

        multifactor = MultiFactor()
        multifactor["name"] = multifactor_name
        multifactor["multifactor_value"] = gather_result
        multifactor["weight"] = "equal_weight"

        return multifactor

    # ic协方差权重加权合成因子---目标:最大化因子IC_IR
    def ic_cov_weighted_factor(self,
                               factor_name_list,
                               factor_value_list,
                               ic_weight_df):
        """
        根据Sample协方差矩阵估算方法得到的因子权重,将若干个因子按该权重加权合成新因子
        :param factor_name_list: 若干因子名称组成的列表(list)
        　　　　                  如：['factor_name_1','factor_name_2',...]
        :param factor_value_list: 若干因子值组成的列表(list)，与factor_name_list一一对应。
                                  每个因子值格式为一个MultiIndex Series，索引(index)为date(level 0)和asset(level 1),
                                  包含一列factor值。
        :param ic_weight_df: 使用Sample 协方差矩阵估算方法得到的因子权重(pd.Dataframe),可通过Admin.get_ic_weight_df 获取。
                             索引（index)为datetime,columns为待合成的因子名称，与factor_name_list一致。
        :return: MultiFactor　对象。包含三个属性：
        　　　　　"name"：合成的因子名称（str)
                "multifactor_value":合成因子值（MultiIndex Series,索引(index)为date(level 0)和asset(level 1),
                                    包含一列factor值)
                "weight": 加权方式 (str)
        """

        import numpy as np
        from utility import MultiFactor

        def strategy_fun(gather, unit):
            return gather + unit

        weight = ic_weight_df
        weighted_factor_value_list = []
        for original_factor in factor_value_list:
            for date in original_factor.index.levels[0]:
                try:
                    weight_by_date = weight[factor_name_list[0]].loc[date]
                    x = original_factor.loc[date] * weight_by_date
                    x.index = pd.MultiIndex.from_product(([date], x.index))
                    original_factor.loc[date] = x
                except:
                    weight_by_date = np.NaN
                    original_factor.loc[date] = original_factor.loc[date] * weight_by_date

            weighted_factor_value_list.append(original_factor)

        # 因子累加
        gather_result = reduce(strategy_fun, weighted_factor_value_list)
        multifactor_name = "+".join(factor_name_list)

        multifactor = MultiFactor()
        multifactor["name"] = multifactor_name
        multifactor["multifactor_value"] = gather_result
        multifactor["weight"] = "ic_cov_weight"

        return multifactor

    # 使用 Ledoit-Wolf 压缩的协方差矩阵估算方法得到的因子权重加权合成因子---目标:最大化因子IC_IR
    def ic_shrink_cov_weighted_factor(self,
                                      factor_name_list,
                                      factor_value_list,
                                      ic_weight_shrink_df):

        """
        根据 Ledoit-Wolf 压缩的协方差矩阵估算方法得到的因子权重,将若干个因子按该权重加权合成新因子
        :param factor_name_list: 若干因子名称组成的列表(list)
        　　　　                  如：['factor_name_1','factor_name_2',...]
        :param factor_value_list: 若干因子值组成的列表(list)，与factor_name_list一一对应。
                                  每个因子值格式为一个MultiIndex Series，索引(index)为date(level 0)和asset(level 1),
                                  包含一列factor值。
        :param ic_weight_shrink_df: 使用Ledoit-Wolf 压缩的协方差矩阵估算方法得到的因子权重(pd.Dataframe),
            　　                    可通过Admin.get_ic_weight_shrink_df 获取。
        　　　　                     索引（index)为datetime,columns为待合成的因子名称，与factor_name_list一致。
        :return: MultiFactor　对象。包含三个属性：
        　　　　　"name"：合成的因子名称（str)
                "multifactor_value":合成因子值（MultiIndex Series,索引(index)为date(level 0)和asset(level 1),
                                    包含一列factor值)
                "weight": 加权方式 (str)
        """

        import numpy as np
        from utility import MultiFactor

        def strategy_fun(gather, unit):
            return gather + unit

        weight = ic_weight_shrink_df
        weighted_factor_value_list = []
        for original_factor in factor_value_list:
            for date in original_factor.index.levels[0]:
                try:
                    weight_by_date = weight[factor_name_list[0]].loc[date]
                    x = original_factor.loc[date] * weight_by_date
                    x.index = pd.MultiIndex.from_product(([date], x.index))
                    original_factor.loc[date] = x
                except:
                    weight_by_date = np.NaN
                    original_factor.loc[date] = original_factor.loc[date] * weight_by_date

            weighted_factor_value_list.append(original_factor)

        # 因子累加
        gather_result = reduce(strategy_fun, weighted_factor_value_list)
        multifactor_name = "+".join(factor_name_list)

        multifactor = MultiFactor()
        multifactor["name"] = multifactor_name
        multifactor["multifactor_value"] = gather_result
        multifactor["weight"] = " ic_shrink_cov_weight"

        return multifactor

    # 获取因子的ic序列
    def get_factors_ic_df(self,
                          factor_name_list,
                          factor_value_list,
                          pool,
                          start,
                          end,
                          periods=(1, 5, 10),
                          quantiles=5,
                          price=None):
        """
        获取指定周期下的多个因子ｉc值序列矩阵
        :param factor_name_list: 若干因子名称组成的列表(list)
        　　　　如：['factor_name_1','factor_name_2',...]
        :param factor_value_list: 若干因子值组成的列表(list)，与factor_name_list一一对应。
               每个因子值格式为一个MultiIndex Series，索引(index)为date(level 0)和asset(level 1),
               包含一列factor值。
        :param pool: 股票池范围（list),如：["000001.XSHE","600300.XSHG",......]
        :param start: 起始时间 (datetime)
        :param end: 结束时间 (datetime)
        :param periods: 指定持有周期（tuple),周期值类型为(int)
        :param quantiles: 根据因子大小将股票池划分的分位数量(int)
        :param price (optional): 包含了pool中所有股票的价格时间序列(pd.Dataframe)，索引（index)为datetime,columns为各股票代码，与pool对应。
        :return: ic_df_dict 指定的不同周期下的多个因子ｉc值序列矩阵所组成的字典(dict), 键为周期（int），值为因子ic值序列矩阵(ic_df)。
                 如：｛１:ic_df_1,5:ic_df_5,10:ic_df_10｝
                 ic_df(ic值序列矩阵) 类型pd.Dataframe，索引（index）为datetime,columns为各因子名称，与factor_name_list对应。
                 如：

                　　　　　　　　　　　BP	　　　CFP	　　　EP	　　ILLIQUIDITY	REVS20	　　　SRMI	　　　VOL20
                date
                2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832	0.214377	0.068445
                2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890	0.202724	0.081748
                2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691	0.122554	0.042489
                2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805	0.053339	0.079592
                2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902	0.077293	-0.050667
        """

        from fxdayu_data import DataAPI
        import datetime
        import numpy as np
        from alphalens import utils, performance

        def get_price_data(pool, start, end, max_window=10):
            data = DataAPI.candle(tuple(pool), "D",
                                  start=start - datetime.timedelta(days=max_window),
                                  end=end + datetime.timedelta(days=max_window))
            data = data.replace(to_replace=0, value=np.NaN)
            return data

        if (price is None):
            price_data = get_price_data(pool.tolist(), start, end, max_window=max(periods))
            price = price_data.minor_xs("close")

        ic_list = []
        for factor_value in factor_value_list:
            # 持股收益-逐只
            stocks_holding_return = utils.get_clean_factor_and_forward_returns(factor_value, price, quantiles=quantiles,
                                                                               periods=periods)
            ic = performance.factor_information_coefficient(stocks_holding_return)
            ic_list.append(ic)

        ic_df_dict = {}
        for period in periods:
            ic_table = []
            for i in range(len(ic_list)):
                ic_by_period = pd.DataFrame(ic_list[i][period])
                ic_by_period.columns = [factor_name_list[i], ]
                ic_table.append(ic_by_period)
            ic_df_dict[period] = pd.concat(ic_table, axis=1)

        return ic_df_dict

    # 样本协方差矩阵估算 - unshrunk covariance
    def get_ic_weight_df(self,
                         ic_df,
                         holding_period,
                         rollback_period=120):
        """
        输入ic_df(ic值序列矩阵),指定持有期和滚动窗口，给出相应的样本协方差矩阵
        :param ic_df: ic值序列矩阵 （pd.Dataframe），索引（index）为datetime,columns为各因子名称。
                 如：

                　　　　　　　　　　　BP	　　　CFP	　　　EP	　　ILLIQUIDITY	REVS20	　　　SRMI	　　　VOL20
                date
                2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832	0.214377	0.068445
                2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890	0.202724	0.081748
                2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691	0.122554	0.042489
                2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805	0.053339	0.079592
                2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902	0.077293	-0.050667

        :param holding_period: 持有周期(int)
        :param rollback_period: 滚动窗口，即计算每一天的因子权重时，使用了之前rollback_period下的IC时间序列来计算IC均值向量和IC协方差矩阵（int)。
        :return: ic_weight_df:使用Sample协方差矩阵估算方法得到的因子权重(pd.Dataframe),
                 索引（index)为datetime,columns为待合成的因子名称。
        """

        import numpy as np

        n = rollback_period
        ic_weight_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns)
        for dt in ic_df.index:
            ic_dt = ic_df[ic_df.index < dt].tail(n)
            if len(ic_dt) < n:
                continue

            ic_cov_mat = np.mat(np.cov(ic_dt.T.as_matrix()).astype(float))
            inv_ic_cov_mat = np.linalg.inv(ic_cov_mat)
            weight = inv_ic_cov_mat * np.mat(ic_dt.mean()).reshape(len(inv_ic_cov_mat), 1)
            weight = np.array(weight.reshape(len(weight), ))[0]
            ic_weight_df.ix[dt] = weight / np.sum(weight)

        return ic_weight_df.shift(holding_period)

    # 样本协方差矩阵估算 - Ledoit-Wolf shrink covariance
    def get_ic_weight_shrink_df(self,
                                ic_df,
                                holding_period,
                                rollback_period=120):
        """
        输入ic_df(ic值序列矩阵),指定持有期和滚动窗口，给出相应的Ledoit-Wolf 压缩方法得到的协方差矩阵估算
        :param ic_df: ic值序列矩阵 （pd.Dataframe），索引（index）为datetime,columns为各因子名称。
                 如：

                　　　　　　　　　　　BP	　　　CFP	　　　EP	　　ILLIQUIDITY	REVS20	　　　SRMI	　　　VOL20
                date
                2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832	0.214377	0.068445
                2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890	0.202724	0.081748
                2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691	0.122554	0.042489
                2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805	0.053339	0.079592
                2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902	0.077293	-0.050667

        :param holding_period: 持有周期(int)
        :param rollback_period: 滚动窗口，即计算每一天的因子权重时，使用了之前rollback_period下的IC时间序列来计算IC均值向量和IC协方差矩阵（int)。
        :return: ic_weight_shrink_df:使用Ledoit-Wolf压缩方法得到的因子权重(pd.Dataframe),
                 索引（index)为datetime,columns为待合成的因子名称。
        """

        from sklearn.covariance import LedoitWolf
        import numpy as np

        n = rollback_period
        ic_weight_shrink_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns)
        lw = LedoitWolf()
        for dt in ic_df.index:
            ic_dt = ic_df[ic_df.index < dt].tail(n)
            if len(ic_dt) < n:
                continue
            ic_cov_mat = lw.fit(ic_dt.as_matrix()).covariance_
            inv_ic_cov_mat = np.linalg.inv(ic_cov_mat)
            weight = inv_ic_cov_mat * np.mat(ic_dt.mean()).reshape(len(inv_ic_cov_mat), 1)
            weight = np.array(weight.reshape(len(weight), ))[0]
            ic_weight_shrink_df.ix[dt] = weight / np.sum(weight)

        return ic_weight_shrink_df.shift(holding_period)

    # 因子间存在较强同质性时，使用施密特正交化方法对因子做正交化处理，用得到的正交化残差作为因子,默认对Admin里加载的所有因子做调整
    def orthogonalize(self,
                      factor_name_list=None,
                      factor_value_list=None,
                      standardize_type="rank"):

        """
        # 因子间存在较强同质性时，使用施密特正交化方法对因子做正交化处理，用得到的正交化残差作为因子,默认对Admin里加载的所有因子做调整
        :param factor_name_list: 若干因子名称组成的列表(list)
        　　　　如：['factor_name_1','factor_name_2',...]
        :param factor_value_list: 若干因子值组成的列表(list)，与factor_name_list一一对应。
               每个因子值格式为一个MultiIndex Series，索引(index)为date(level 0)和asset(level 1),
               包含一列factor值。
        :param standardize_type: 标准化方法，有"rank"（排序标准化）,"z_score"(z-score标准化)两种（"rank"/"z_score"）
        :return: factor_name_list（new),factor_value_list(new) 正交化处理后所得的新因子名称列表，因子值列表。
        """

        from scipy import linalg
        from factor import Factor

        def Schmidt(data):
            return linalg.orth(data)

        def get_vector(date, factor):
            return factor.loc[date]

        if not factor_name_list or not factor_value_list:
            if not self._all_factors_value:
                raise TypeError("There is no factor calculated.")
            else:
                factor_name_list = list(self._all_factors_value.keys())
                factor_value_list = list(self._all_factors_value.values())

        if len(factor_name_list) < 2:
            raise TypeError("you must give more than 2 factors.")

        factor_value_dict = {}  # 用于记录正交化后的因子值
        for factor_name in factor_name_list:
            factor_value_dict[factor_name] = []

        # 施密特正交
        for date in factor_value_list[0].index.levels[0]:
            data = map(partial(get_vector, date), factor_value_list)
            data = pd.concat(data, axis=1, join="inner")
            if len(data) == 0:
                continue
            if pd.isnull(data).values.any():
                continue
            data = pd.DataFrame(Schmidt(data), index=data.index)
            data.columns = factor_name_list
            for factor_name in data.columns:
                row = pd.DataFrame(data[factor_name]).T
                row.index = [date, ]
                factor_value_dict[factor_name].append(row)

        # 因子标准化
        for factor_name in factor_name_list:
            factor_value = pd.concat(factor_value_dict[factor_name])
            if standardize_type == "z_score":
                factor_value = Factor.standardize(factor_value)
                factor_value_dict[factor_name] = Factor.factor_df_to_factor_mi(factor_value)
            else:
                factor_value = Factor.factor_df_to_factor_mi(factor_value)
                factor_value_dict[factor_name] = Factor.get_factor_by_rankScore(factor_value)

        return list(factor_value_dict.keys()), list(factor_value_dict.values())

    @staticmethod
    def calculate_performance(factor_name,
                              factor_value,
                              start,
                              end,
                              periods=(1, 5, 10),
                              quantiles=5,
                              price=None
                              ):

        """
        # 计算因子（含组合因子）的表现
        :param factor_name: 因子名称 string
        :param factor_value: 因子值　格式为一个MultiIndex Series，索引(index)为date(level 0)和asset(level 1),
               包含一列factor值。
        :param start: 回测起始时间 datetime
        :param end: 回测结束时间 datetime
        :param periods: 持有时间 tuple
        :param quantiles: 划分分位数 int
        :param price （optional）:计算绩效时用到的的个股每日价格,通常为收盘价（close）。索引（index)为datetime,columns为各股票代码。pandas dataframe类型,形如:
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

        :return:该因子的表现 (Performance object)
                包含"factor_name", "holding_return", "mean_return_by_q", "ic", "mean_ic_by_M", "mean_ic"这些属性。

        """
        from utility import Performance
        from fxdayu_data import DataAPI
        import alphalens
        import numpy as np
        import datetime

        def get_price_data(pool, start, end, max_window=10):
            data = DataAPI.candle(tuple(pool), "D",
                                  start=start - datetime.timedelta(days=max_window),
                                  end=end + datetime.timedelta(days=max_window))
            data = data.replace(to_replace=0, value=np.NaN)
            return data

        # 判断是否结果为空,为空则返回空字典
        if len(factor_value) == 0:
            return None

        pool = factor_value.index.levels[1]

        if (price is None):
            price_data = get_price_data(pool.tolist(), start, end, max_window=max(periods))
            price = price_data.minor_xs("close")

        ################################################################################
        # 以下计算各绩效指标
        performance = Performance()

        performance["factor_name"] = factor_name

        # 持有期收益
        performance["holding_return"] = alphalens.utils.get_clean_factor_and_forward_returns(factor_value,
                                                                                             price,
                                                                                             quantiles=quantiles,
                                                                                             periods=periods)

        # 按quantile区分的持股平均收益（减去了总体平均值）
        performance["mean_return_by_q"] = \
            alphalens.performance.mean_return_by_quantile(performance["holding_return"], by_date=True, demeaned=True)[0]

        # 因子的IC值
        performance["ic"] = alphalens.performance.factor_information_coefficient(performance["holding_return"])

        # 平均IC值-月
        performance["mean_ic_by_M"] = alphalens.performance.mean_information_coefficient(performance["holding_return"],
                                                                                         by_time="M")

        # 总平均IC值
        performance["mean_ic"] = alphalens.performance.mean_information_coefficient(performance["holding_return"])

        return performance

    def rank_performance(self,
                         factors_performance,
                         target_period=10,
                         ascending=False
                         ):
        """
        将若干Performance对象所组成的列表（factors_performance）按指定持有期(target_period)下的"mean_ic"排序，默认为降序
        :param factors_performance: 若干Performance对象所组成的列表（list）
               Performance object 包含"factor_name", "holding_return", "mean_return_by_q", "ic", "mean_ic_by_M", "mean_ic"这些属性。
        :param target_period: 指定持有期(int)
        :param ascending: 是否升序（bool）。默认False(降序）
        :return: 排序后的Performance对象所组成的列表
        """

        return sorted(factors_performance,
                      key=lambda x: x.mean_ic.loc[target_period].values[0],
                      reverse=(ascending == False))

    # 遍历list中各因子的结果，计算其绩效表现，并汇总成列表——list
    def show_factors_performance(self,
                                 factor_name_list,
                                 factor_value_list,
                                 start,
                                 end,
                                 periods=(1, 5, 10),
                                 quantiles=5,
                                 price=None,
                                 parallel=False):
        """
        批量计算factor_name_list中所有因子的表现。
        :param factor_name_list: 若干因子名称组成的列表(list)
        　　　　如：['factor_name_1','factor_name_2',...]
        :param factor_value_list: 若干因子值组成的列表(list)，与factor_name_list一一对应。
               每个因子值格式为一个MultiIndex Series，索引(index)为date(level 0)和asset(level 1),
               包含一列factor值。
        :param start: 回测起始时间 datetime
        :param end: 回测结束时间 datetime
        :param periods: 持有时间 tuple
        :param quantiles: 划分分位数 int
        :param price （optional）:计算绩效时用到的的个股每日价格,通常为收盘价（close）。索引（index)为datetime,columns为各股票代码。pandas dataframe类型,形如:
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
        :param parallel: 是否执行并行计算（bool） 默认不执行。 如需并行计算需要在ipython notebook下启动工作脚本。
        :return:因子的表现 (Performance object)所组成的列表(list),
                列表里每个元素为因子的表现 (Performance object)
                包含"factor_name", "holding_return", "mean_return_by_q", "ic", "mean_ic_by_M", "mean_ic"这些属性。
        """

        if parallel:
            client = Client()
            lview = client.load_balanced_view()
            results = []
            for i in range(len(factor_name_list)):
                results.append(lview.apply_async(self.calculate_performance,
                                                 factor_name_list[i],
                                                 factor_value_list[i],
                                                 start,
                                                 end,
                                                 periods=periods,
                                                 quantiles=quantiles,
                                                 price=price))
            lview.wait(results)
            factors_performance = [result.get() for result in results if not (result.get() == None)]
            return factors_performance
        else:
            factors_performance = []
            for i in range(len(factor_name_list)):
                result = _apply(self.calculate_performance,
                                factor_name_list[i],
                                factor_value_list[i],
                                start,
                                end,
                                periods=periods,
                                quantiles=quantiles,
                                price=price)
                if not (result == None):
                    factors_performance.append(result)
            return factors_performance

    @staticmethod
    # 计算某个因子指定时间段的因子值
    def instantiate_factor_and_get_factor_value(factor_name,
                                                pool,
                                                start,
                                                end,
                                                Factor=None,
                                                data=None,
                                                data_config={"freq": "D", "api": "candle", "adjust": "after"},
                                                para_dict=None,
                                                ):
        """
        计算某个因子指定时间段的因子值
        :param factor_name: 因子名称（str） 需确保传入的factor_name、因子的类名、对应的module文件名一致(不含.后缀),因子才能正确加载
        :param pool: 股票池范围（list),如：["000001.XSHE","600300.XSHG",......]
        :param start: 起始时间 (datetime)
        :param end: 结束时间 (datetime)
        :param Factor (optional): 因子(factor.factor.Factor object),可选.可以输入一个设计好的Factor类来执行计算.
        :param data (optional): 计算因子需用到的数据,根据计算需求自行指定。(可选)
        :param data_config (optional): 在data参数为None的情况下(不传入自定义数据),
                                       可通过该参数调用dxdayu_data api 访问到数据 (dict),
                                       与data参数二选一。
        :param para_dict (optional): 外部指定因子里所用到的参数集(dict),为空则不修改原有参数。 形如:{"fast":5,"slow":10}
        :return: factor_value:因子值　格式为一个MultiIndex Series，索引(index)为date(level 0)和asset(level 1),
               包含一列factor值。形如：
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

        from fxdayu_data import DataAPI
        import datetime

        # 实例化因子类
        if Factor is None:
            factor = _get_factor(factor_name, Admin.PACKAGE_NAME)()
        else:
            factor = Factor

        # 接收外部传入的参数
        if para_dict:
            for para in para_dict.keys():
                setattr(factor, para, para_dict[para])

        if data is None:
            pn_data = DataAPI.get(symbols=tuple(pool),
                                  start=start - datetime.timedelta(days=factor.max_window),
                                  end=end,
                                  **data_config)
        else:
            pn_data = data

        # 因子计算结果获取
        factor_value = factor.factor(pn_data, update=True)

        return factor_value

    # 因子参数枚举器
    def enumerate_parameter(self,
                            factor_name,
                            para_range_dict,
                            pool,
                            start,
                            end,
                            Factor=None,
                            data=None,
                            data_config={"freq": "D", "api": "candle", "adjust": "after"},
                            parallel=False):
        """
        # 枚举不同参数下的所得到的不同因子值
        :param factor_name: 因子名称（str） 需确保传入的factor_name、因子的类名、对应的module文件名一致(不含.后缀),因子才能正确加载
        :param para_range_dict: 述了factor当中待优化参数的选择空间（dict）。键为参数名称，值为range对象，表示优化空间的起始、终止、步长。
               如：para_range_dict = {“fast”：range(0,10,1),"slow":range(0,10,1)}.
        :param pool: 股票池范围（list),如：["000001.XSHE","600300.XSHG",......]
        :param start: 起始时间 (datetime)
        :param end: 结束时间 (datetime)
        :param Factor (optional): 因子(factor.factor.Factor object),可选.可以输入一个设计好的Factor类来执行计算.
        :param data (optional): 计算因子需用到的数据,根据计算需求自行指定。(可选)
        :param data_config (optional): 在data参数为None的情况下(不传入自定义数据),可通过该参数调用dxdayu_data api 访问到数据 (dict)
        :param factor_package_name: 因子所在的package名称 (str)
        :param parallel: 是否执行并行计算（bool） 默认不执行。 如需并行计算需要在ipython notebook下启动工作脚本。
        :return: factor_value_by_para_list, para_dict_list
                 factor_value_by_para_list：不同参数下得到的因子值所组成的list（list）
                 para_dict_list：不同参数集所组成的list（list），与factor_value_by_para_list一一对应。
                                 每个参数集格式为dict，形如:{"fast":5,"slow":10}
        """

        from itertools import product

        keys = para_range_dict.keys()

        if parallel:
            client = Client()
            lview = client.load_balanced_view()
            results = []
            para_dict_list = []
            for value in product(*para_range_dict.values()):
                para_dict = dict(zip(keys, value))
                para_dict_list.append(para_dict)
                results.append(lview.apply_async(self.instantiate_factor_and_get_factor_value,
                                                 factor_name,
                                                 pool,
                                                 start,
                                                 end,
                                                 Factor,
                                                 data,
                                                 data_config,
                                                 para_dict))

            lview.wait(results)
            diff_results = [result.get() for result in results]
            return diff_results, para_dict_list
        else:
            results = []
            para_dict_list = []
            for value in product(*para_range_dict.values()):
                para_dict = dict(zip(keys, value))
                para_dict_list.append(para_dict)
                results.append(_apply(self.instantiate_factor_and_get_factor_value,
                                      factor_name,
                                      pool,
                                      start,
                                      end,
                                      Factor,
                                      data,
                                      data_config,
                                      para_dict))
            return results, para_dict_list

    # 获取admin下所有因子的计算结果
    def get_all_factors_value(self,
                              pool,
                              start,
                              end,
                              all_Factors_dict=None,
                              all_factors_data_dict=None,
                              all_factors_data_config_dict=None,
                              all_factors_para_dict=None,
                              factor_package_name="factors",
                              parallel=False,
                              update=False):
        """
        计算admin下加载的所有因子的因子值

        :param pool: 股票池范围（list),如：["000001.XSHE","600300.XSHG",......]
        :param start: 起始时间 (datetime)
        :param end: 结束时间 (datetime)
        :param all_Factors (optional):加载到admin下的所有因子类(factor.factor.Factor object)构成的字典, 可选.
                                      可以输入一系列设计好的Factor类(与Admin._all_factors_name一一对应)直接执行计算.
                                      形如:{“factor_name_1”:Factor_1,factor_name_2”:Factor_2,...}
        :param all_factors_data_dict （optional): 计算因子需用到的自定义数据组成的字典（dict）,根据计算需求自行指定。
                                                  字典键名为所有载入的因子的因子名(admin._all_factors_name),值为对应因子所需的数据。
                                                  形如：{“factor_name_1”:data_1,factor_name_2”:data_2,...}
        :param all_factors_data_config_dict (optional):  在all_factors_data_dict参数为None的情况下(不传入自定义数据),
                                                         可通过该参数调用dxdayu_data api 访问到数据 (dict).
                                                         与 all_factors_data_dict 二选一（未指定数据通过fxdayu_data api获取）.
                                                         字典键名为所有载入的因子的因子名(admin._all_factors_name),
                                                         值为对应因子所需的数据api访问参数设置dict(data_config)。
                                                         形如：{“factor_name_1”:data_config_1,factor_name_2”:data_config_2,...}
        :param all_factors_para_dict (optional): 所有因子外部指定参数集(dict)所构成的字典(dict),可选。为空则不修改因子原有参数。
                                                 字典键名为所有载入的因子的因子名(admin._all_factors_name),
                                                 值为对应因子的指定参数集(dict)。
                                                 形如: {“factor_name_1”:{"fast":5,"slow":10},factor_name_2”:{"fast":4,"slow":7},...}
        :param factor_package_name: 因子所在的package名称 (str)
        :param parallel: 是否执行并行计算（bool） 默认不执行。 如需并行计算需要在ipython notebook下启动工作脚本。
        :param update: 是否更新已有记录值(bool)。默认为False——如果admin曾经计算过所有因子值,则不再重复计算。 True 则更新计算所加载的因子值。
        :return: all_factors_value : admin下加载的所有因子的因子值(dict)。
                                     字典键名为所有载入的因子的因子名(admin._all_factors_name)
                                     值为 因子值(factor_value)　格式为一个MultiIndex Series，索引(index)为date(level 0)和asset(level 1),
                                     包含一列factor值。
        """

        if self._all_factors_value and not update:
            return self._all_factors_value
        else:
            # data参数验证
            if not all_factors_data_dict:
                all_factors_data_dict = {}
                for factor_name in self._all_factors_name:
                    all_factors_data_dict[factor_name] = None
            if not all_factors_data_config_dict:
                all_factors_data_config_dict = {}
                for factor_name in self._all_factors_name:
                    all_factors_data_config_dict[factor_name] = {"freq": "D", "api": "candle", "adjust": "after"}
            if not all_factors_para_dict:
                all_factors_para_dict = {}
                for factor_name in self._all_factors_name:
                    all_factors_para_dict[factor_name] = None
            if not all_Factors_dict:
                all_Factors_dict = {}
                for factor_name in self._all_factors_name:
                    all_Factors_dict[factor_name] = None

            self._all_factors_value = {}
            if parallel:
                client = Client()
                lview = client.load_balanced_view()
                factors_value = {}
                for factor_name in self._all_factors_name:
                    factors_value[factor_name] = lview.apply_async(self.instantiate_factor_and_get_factor_value,
                                                                   factor_name,
                                                                   pool,
                                                                   start,
                                                                   end,
                                                                   all_Factors_dict[factor_name],
                                                                   all_factors_data_dict[factor_name],
                                                                   all_factors_data_config_dict[factor_name],
                                                                   all_factors_para_dict[factor_name])
                lview.wait(factors_value.values())
                self._all_factors_value = {name: result.get() for name, result in factors_value.items()}
            else:

                for factor_name in self._all_factors_name:
                    self._all_factors_value[factor_name] = _apply(self.instantiate_factor_and_get_factor_value,
                                                                  factor_name,
                                                                  pool,
                                                                  start,
                                                                  end,
                                                                  all_Factors_dict[factor_name],
                                                                  all_factors_data_dict[factor_name],
                                                                  all_factors_data_config_dict[factor_name],
                                                                  all_factors_para_dict[factor_name])
            return self._all_factors_value
