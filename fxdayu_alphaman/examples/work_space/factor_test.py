# -*- coding: utf-8 -*-

#######################################################
#
# 因子器基本功能测试 以volume001为例
#
########################################################


import datetime
import json

from fxdayu_data import DataAPI

from fxdayu_alphaman.factor.admin import Admin
from fxdayu_alphaman.examples.factors.Factor_Volume001 import Factor_Volume001
from fxdayu_alphaman.factor.utility import standard_code_style


# 配置选股器所在包路径
Admin.PACKAGE_NAME = "examples.factors"

# 初始选股范围设置
initial_codes = standard_code_style(json.load(open('test_stock_pool.json'))["test_stock_pool"])
data_config = {"freq": "D", "api": "candle", "adjust": "after"}

# 测试参数设置
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2016, 4, 18, 15)
periods = (1, 5, 10)

# 获取数据
data = DataAPI.get(symbols=tuple(initial_codes),
                   start=start - datetime.timedelta(days=100),
                   end=end,
                   **data_config)

prices = data.minor_xs("close")


def unit_test1(data):
    volume001 = Factor_Volume001()
    factor = volume001.factor(data)
    return factor


def test_performance(factor, prices):
    import matplotlib.pyplot as plt
    from alphalens import utils, performance, plotting

    # 持股收益-逐只
    stocks_holding_return = utils.get_clean_factor_and_forward_returns(factor, prices, quantiles=5, periods=(1, 5, 10))

    print("因子的IC值：")
    ic = performance.factor_information_coefficient(stocks_holding_return)
    print(ic)
    plotting.plot_ic_hist(ic)
    plt.show()
    plotting.plot_ic_ts(ic)
    plt.show()

    print("平均IC值-月：")
    mean_ic = performance.mean_information_coefficient(stocks_holding_return, by_time="M")
    plotting.plot_monthly_ic_heatmap(mean_ic)
    plt.show()

    # 按quantile区分的持股平均收益（减去了总体平均值）
    mean_return_by_q = performance.mean_return_by_quantile(stocks_holding_return, by_date=True, demeaned=True)[0]
    # 按quantile画出累积持有收益
    for i in [1, 5, 10]:
        plotting.plot_cumulative_returns_by_quantile(mean_return_by_q, period=i)
        plt.show()


factor = unit_test1(data)
test_performance(factor, prices)
#
# # 参数优化
admin = Admin()
original_perf = admin.calculate_performance("Factor_Volume001", factor, start, end, periods=(1, 5, 10), quantiles=5,
                                            price=prices)
print(original_perf.mean_ic)  # 以前的绩效－ic

para_range_dict = {"c": range(4, 11, 1)}
factor_value_list, para_dict_list = admin.enumerate_parameter("Factor_Volume001", para_range_dict, initial_codes, start,
                                                              end,
                                                              data_config=data_config)

factor_name_list = []
for para_dict in para_dict_list:
    factor_name_list.append("Factor_Volume001+" + str(para_dict))

performance_list = admin.show_factors_performance(factor_name_list, factor_value_list, start, end, periods=(1, 5, 10),
                                                  quantiles=5, price=prices)

print("#####################################################################################")
# 按绩效指标对结果排序（寻优） 本例按10天持有期的mean_IC降序排列了所有结果。
performance_list = admin.rank_performance(performance_list,
                                          target_period=10,
                                          ascending=False)

for perf in performance_list:
    print("\n")
    print(perf.mean_ic)
    print(perf.factor_name)
