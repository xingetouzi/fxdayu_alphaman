# -*- coding: utf-8 -*-

#######################################################
#
# 选股器基本功能测试 以MA为例
#
########################################################

import json
from datetime import datetime

from fxdayu_alphaman.selector.selector_analysis import *
from fxdayu_alphaman.selector.utility import read_benchmark, standard_code_style
from fxdayu_alphaman.examples.selectors.DayMA import DayMA
from fxdayu_alphaman.selector.admin import Admin


# 配置选股器所在包路径
Admin.PACKAGE_NAME = "fxdayu_alphaman.examples.selectors"

# 初始选股范围设置
initial_codes = standard_code_style(json.load(open('test_stock_pool.json'))["test_stock_pool"])
data_config = {"freq": "D", "api": "candle", "adjust": "after"}

# 测试参数设置
start = datetime(2016, 1, 1)
end = datetime(2017, 4, 18, 15)
periods = (1, 5, 10)

# benchmark读取
hs300 = read_benchmark(start, end)
hs300_return = get_stocklist_mean_return(hs300["index"], "hs300", start, end, hs300["close"], periods=periods)


def unit_test1():
    # 选股器单元测试1
    selector = DayMA()
    result = selector.selector_result(pool=initial_codes,
                                      start=start,
                                      end=end,
                                      data_config=data_config)
    print(result)  # 输出选股结果
    return result


def unit_test2():
    # 选股器单元测试2——通过Admin
    selector_admin = Admin()
    result = selector_admin.instantiate_selector_and_get_selector_result([], initial_codes, start, end,
                                                                         Selector=DayMA(),
                                                                         data_config=data_config)
    print(result)
    return selector_admin, result


def test_performance(selector_admin, result):
    # 选股绩效测试
    performance = selector_admin.calculate_performance("DayMA",
                                                       result[result > 0],  # 结果大于0的（选出的）
                                                       start,
                                                       end,
                                                       periods=periods,
                                                       benchmark_return=hs300_return)
    print(performance["mean_return"])  # 选股策略平均持有收益
    print(performance["key_performance_indicator"])  # 关键性绩效指标
    print(performance["upside_return"])  # 上行收益
    print(performance["upside_distribution_features"])  # 上行收益特征
    print(performance["downside_return"])  # 下行收益
    print(performance["downside_distribution_features"])  # 下行收益特征

    # 支持类属性的调用方式
    print(performance.holding_return)  # 持有收益
    print(performance.holding_distribution_features)  # 持有收益特征

    return performance


def test_plotting(performance):
    # 画图测试

    # 1.收益概率密度分布图
    plot_distribution_of_returns(performance["upside_return"], period=1, return_type="upside")
    plt.show()
    # 2.收益概率密度分布图-提琴盒图
    plot_stock_returns_violin(performance["downside_return"], return_type="downside")
    plt.show()
    # 3.累积收益曲线
    plot_cumulative_returns_by_quantile(performance["mean_return"], period=10)
    plt.show()


####### test1
unit_test1()
####### test2
selector_admin, result = unit_test2()
#######
performance = test_performance(selector_admin, result)
#######
test_plotting(performance)

#######
# 优化选股器参数
para_range_dict = {"fast": range(2, 5, 1), "slow": range(10, 15, 1)}
results_list, para_dict_list = selector_admin.enumerate_parameter("DayMA",
                                                                  para_range_dict,
                                                                  initial_codes,
                                                                  start,
                                                                  end,
                                                                  data_config=data_config,
                                                                  parallel=False)

# 批量计算多个不同参数下方案的绩效表现

strategy_name_list = []
strategy_result_list = []
for i in range(len(results_list)):
    print("\n")
    strategy_name = "DayMA+" + str(para_dict_list[i])
    print(strategy_name)
    strategy_name_list.append(strategy_name)

    strategy_result = results_list[i][results_list[i] > 0]
    print(strategy_result)
    strategy_result_list.append(strategy_result)  # 只记录大于0的结果

performance_list = selector_admin.show_strategies_performance(strategy_name_list,
                                                              strategy_result_list,
                                                              start,
                                                              end,
                                                              periods=periods,
                                                              benchmark_return=hs300_return,
                                                              parallel=False)

for perf in performance_list:
    print("\n")
    print(perf.strategy_name)
    print(perf.key_performance_indicator)

print("#####################################################################################")
# 按绩效指标对结果排序（寻优） 本例按10天持有期的夏普比降序排列了所有结果。
performance_list = selector_admin.rank_performance(performance_list,
                                                   target_period=10,
                                                   target_indicator="Sharpe ratio",
                                                   ascending=False)

for perf in performance_list:
    print("\n")
    print(perf.strategy_name)
    print(perf.key_performance_indicator)

# 画出排序最靠前的绩效曲线
# 1.收益概率密度分布图
plot_distribution_of_returns(performance_list[0]["upside_return"], period=10, return_type="upside")
plt.show()
# 2.收益概率密度分布图-提琴盒图
plot_stock_returns_violin(performance_list[0]["downside_return"], return_type="downside")
plt.show()
# 3.累积收益曲线
plot_cumulative_returns_by_quantile(performance_list[0]["mean_return"], period=10)
plt.show()
