# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from fxdayu_alphaman.factor.factor import Factor


class Factor_Volume003(Factor):

    d = 7
    s = 8
    c = 14

    def calculate_volume003(self, data):
        # 逐支股票计算volume003因子
        candle_data = data[1].dropna()
        if len(candle_data) == 0:
            return
        high = candle_data["high"]
        volume = candle_data["volume"]
        factor_volume003 = - self.correlation(self.slope(high, self.s), self.slope(self.delta(np.log(volume), self.d), self.s),self.c) #计算因子值
        factor_volume003.index = candle_data.index
        factor_volume003 = pd.DataFrame(factor_volume003)
        factor_volume003.columns = [data[0],]
        return  factor_volume003

    def factor_calculator(self, pn_data):
        # volume003

        factor_volume003 = list(map(self.calculate_volume003, pn_data.iteritems()))
        factor_volume003 = pd.concat(factor_volume003, axis=1)
        factor_volume003 = self.winsorize(factor_volume003) #去极值
        factor_volume003 = self.standardize(factor_volume003) #标准化
        #factor_volume003 = self.neutralize(factor_volume003, factorIsMV=False) #行业、市值中性化
        factor_volume003 = self.factor_df_to_factor_mi(factor_volume003)  #转multiIndex格式
        factor_volume003 = self.get_factor_by_rankScore(factor_volume003, ascending=True) # 将因子用排序分值重构，并处理到0-1之间(默认为升序——因子越大 排序分值越大(越好)
                                                                                          # 具体根据因子对收益的相关关系而定，为正则应用升序,为负用降序)
        factor_volume003 = self.get_disturbed_factor(factor_volume003)  # 将因子值加一个极小的扰动项,
                                                                        # 避免过多因子值相同而没有区分的情况.
        return factor_volume003
