# -*- coding: utf-8 -*-

#######################################################
#
# factor admin组合功能测试
#
########################################################


import datetime
import json
from fxdayu_data import DataAPI
from fxdayu_alphaman.factor.admin import Admin

from fxdayu_alphaman.factor.utility import standard_code_style

# 配置选股器所在包路径
Admin.PACKAGE_NAME = "fxdayu_alphaman.examples.factors"

# 初始选股范围设置
initial_codes = standard_code_style(json.load(open('test_stock_pool.json'))["test_stock_pool"])
data_config = {"freq": "D", "api": "candle", "adjust": "after"}

# 测试参数设置
start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2017, 4, 18, 15)
periods = (1, 5, 10)

# 获取数据
data = DataAPI.get(symbols=tuple(initial_codes),
                   start=start - datetime.timedelta(days=100),
                   end=end,
                   **data_config)

prices = data.minor_xs("close")


def manage_factors_value_test(factor_name_list, data_config_dict):
    # admin测试 -获得多个因子结果
    factor_admin = Admin(*factor_name_list)
    result = factor_admin.get_all_factors_value(initial_codes, start, end,
                                                all_factors_data_config_dict=data_config_dict)

    return factor_admin, result


#######################################################
# 确定要载入的因子名称
factor_name_list = ["Factor_Volume003", "Factor_Volume001"]
# 逐个配置因子需要的数据类型
data_config_dict = {"Factor_Volume003": data_config, "Factor_Volume001": data_config}

# admin测试 -获得多个因子结果
factor_admin, result = manage_factors_value_test(factor_name_list, data_config_dict)

#######################################

factor_name_list, factor_value_list = list(result.keys()), list(result.values())

# 因子加权合成

# 1)　计算因子ｉｃ序列
ic_df = factor_admin.get_factors_ic_df(factor_name_list, factor_value_list, initial_codes, start, end,
                                       periods=(1, 5, 10), quantiles=5, price=prices)
# 2) 计算因子权重
holding_period = 10
ic_weight_df = factor_admin.get_ic_weight_df(ic_df[holding_period], holding_period, rollback_period=30)

# 3)计算加权合成的因子
new_factor = factor_admin.ic_cov_weighted_factor(factor_name_list, factor_value_list, ic_weight_df)

# 4)查看合成的因子表现
perf = factor_admin.calculate_performance(new_factor.name, new_factor.multifactor_value, start, end, periods=(1, 5, 10),
                                          quantiles=5, price=prices)
print(perf.mean_ic)

from alphalens import plotting
import matplotlib.pyplot as plt

plotting.plot_ic_hist(perf.ic)
plt.show()
plotting.plot_ic_ts(perf.ic)
plt.show()
plotting.plot_monthly_ic_heatmap(perf.mean_ic_by_M)
plt.show()

# 按quantile画出累积持有收益
for i in [1, 5, 10]:
    plotting.plot_cumulative_returns_by_quantile(perf.mean_return_by_q, period=i)
    plt.show()
