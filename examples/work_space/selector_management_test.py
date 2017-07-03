# -*- coding: utf-8 -*-

#######################################################
#
# 选股器admin组合功能测试
#
########################################################

import json
from datetime import datetime
from fxdayu_alphaman.selector.selector_analysis import *
from fxdayu_alphaman.selector.utility import read_benchmark, standard_code_style

from fxdayu_alphaman.selector.admin import Admin

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


def manage_selector_result_test(selector_name_list, data_config_dict):
    # admin测试 -获得多个选股器结果
    selector_admin = Admin(*selector_name_list)
    result = selector_admin.get_all_selectors_result(initial_codes,
                                                     start,
                                                     end,
                                                     all_selectors_data_config_dict=data_config_dict,
                                                     parallel=False)

    return selector_admin, result


#######################################################
# 配置选股器所在包路径
Admin.PACKAGE_NAME = "examples.selectors"

# 确定要载入的选股器名称
selector_name_list = ["DayMA", "DayMACD", "Volume003"]

# 逐个配置选股器需要的数据类型
data_config_dict = {"DayMA": data_config, "Volume003": data_config, "DayMACD": data_config}

# admin测试 -获得多个选股器结果
selector_admin, result = manage_selector_result_test(selector_name_list, data_config_dict)

#######################################
# # adimin测试 - 选股组合
selector_result_list = []
for selector_name in selector_name_list:
    choice = result[selector_name]
    selector_result_list.append(choice)

# #1.选股结果取并集
Union_Strategy = selector_admin.Union_Strategy(selector_name_list, selector_result_list)
print(Union_Strategy.strategy_result)
#
# #2.取交集
Intersection_Strategy = selector_admin.Intersection_Strategy(selector_name_list, selector_result_list)
print(Intersection_Strategy.strategy_result)

# 3.给不同选股器配权重 组合打分 并取分值大的
weight_dict = {"DayMA": 1, "Volume003": 5, "DayMACD": 3}
Rank_Strategy = selector_admin.Rank_Strategy(selector_name_list, selector_result_list, rank=10, weight_dict=weight_dict)
print(Rank_Strategy.strategy_result)

# 4.枚举不同的选股方案组合
selector_name_lists = selector_admin.max_combination(alist=selector_name_list, max_order=3)  # 获取可能的所有组合情况
strategies = selector_admin.combinate_selectors_result(selector_admin.Union_Strategy,
                                                       selector_name_lists,
                                                       parallel=False)

for strategy in strategies:
    print("\n")
    print(strategy.strategy_name)
    print(strategy.strategy_result)

# #5. 枚举不同的选股器权重
weight_range_dict = {"DayMA": range(0, 2, 1), "DayMACD": range(0, 2, 1), "Volume003": range(0, 2, 1)}
weighted_strategies = selector_admin.enumerate_selectors_weight(selector_admin.Rank_Strategy,
                                                                weight_range_dict,
                                                                selector_name_list,
                                                                rank=10,
                                                                parallel=False)

for strategy in weighted_strategies:
    print(strategy.strategy_name)
    print(strategy.weight_dict)
    print(strategy.strategy_result)

# 6.批量计算多个选股方案（含选股组合方案）绩效表现
strategy_name_list = []
strategy_result_list = []
for strategy in strategies:
    strategy_name_list.append(strategy.strategy_name + "+" + str(strategy.weight_dict))
    strategy_result_list.append(strategy.strategy_result)

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
# 7. 按绩效指标对结果排序（寻优） 本例按10天持有期的夏普比降序排列了所有结果。
performance_list = selector_admin.rank_performance(performance_list,
                                                   target_period=10,
                                                   target_indicator="sharpe_ratio",
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
