# -*- coding: utf-8 -*-

import pandas as pd

from fxdayu_alphaman.factor.factor import Factor


class Factor_Volume001(Factor):

    c = 4

    def calculate_volume001(self, data):
        # 逐支股票计算volume003因子
        candle_data = data[1].dropna()
        if len(candle_data) == 0:
            return
        high = candle_data["high"]
        volume = candle_data["volume"]
        adv_s = self.ts_mean(volume, 10)
        factor_volume001 = - self.correlation(high, adv_s, self.c) #计算因子值
        factor_volume001.index = candle_data.index
        factor_volume001 = pd.DataFrame(factor_volume001)
        factor_volume001.columns = [data[0],]
        return  factor_volume001

    def factor_calculator(self, pn_data):
        # volume001

        factor_volume001 = list(map(self.calculate_volume001, pn_data.iteritems()))
        factor_volume001 = pd.concat(factor_volume001, axis=1)
        factor_volume001 = self.winsorize(factor_volume001) #去极值
        factor_volume001 = self.standardize(factor_volume001) #标准化
        factor_volume001 = self.neutralize(factor_volume001, factorIsMV=False) #行业、市值中性化
        factor_volume001 = self.factor_df_to_factor_mi(factor_volume001) #格式标准化
        factor_volume001 = self.get_factor_by_rankScore(factor_volume001, ascending=True) # 将因子用排序分值重构，并处理到0-1之间(默认为升序——因子越大 排序分值越大(越好)
                                                                                          # 具体根据因子对收益的相关关系而定，为正则应用升序,为负用降序)
        return factor_volume001
