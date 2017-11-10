from functools import partial

import numpy as np
from sklearn.preprocessing import minmax_scale, scale, robust_scale, maxabs_scale


def minus_median(array):
    return array - np.median(array, axis=0)


class Basic(object):
    def __init__(self, PN):
        """

        :param PN: the axes of input panel are [factors, time, stocks] 
                    and the last column of factor should be 'ret';
        """
        self.pn = PN
        self.prep_methods = {
            "scale": scale,
            "minmax": minmax_scale,
            "robust": robust_scale,
            "maxabs": maxabs_scale,
            "maxabs": maxabs_scale,
            "minus_median": minus_median
        }
        print("Axes of input panel should be [factors, time, stocks], \n"
              "and the last column of factor should be 'ret', \n"
              " pn = pn.transpose(2,1,0)")

    def prep(self, method, specific=None, **kwargs):

        """
        设定只对横截面上的特征值进行处理，对目标值不做处理
        input: [factors, time, stocks]
        :param method: str，如 "scale", "minmax", "robust", "maxabs", "minus_median"
        :param specific: str or list of str 因子的名字，指定特定的factors做处理，可以为None
        :return: 
        """
        # the original axes of panel are [factors, time, stocks]]
        df_name = self.pn.axes[0][-1]
        tmp_df, self.pn = self.pn.iloc[-1, :, :], self.pn.iloc[:-1, :, :]

        func = partial(self.prep_methods[method], **kwargs)

        if specific is None:

            self.pn = self.pn.to_frame().groupby(level=0).transform(func).to_panel()

        else:

            tmp = self.pn.to_frame()
            tmp[:, specific] = tmp[:, specific].groupby(level=0).transform(func)
            self.pn = tmp.to_panel()

        self.pn[df_name] = tmp_df

    def add_prep_methods(self, name, func):
        """
        添加做预处理的方法
        :param name: name of the input func 
        :param func: 形式如function(DataFrame), DataFrame的index，columns分别为stocks, factors
        :return: 
        """
        if name not in self.prep_methods:
            self.prep_methods[name] = func

    def eval(self, operations=None, **kwargs):
        """
        在横截面上添加特征的方法, 不支持全局变量的加入，要加入全局变量直接通过self.pn进行添加
        :param operations: list, 数学表达式的列表，形式如['c=a+b'] 
        """
        # factor, time, stocks
        if operations:
            if not isinstance(operations, list):
                raise TypeError("operations should be a list")
            else:
                df_name = self.pn.axes[0][-1]
                target_df, self.pn = self.pn.iloc[-1, :, :], self.pn.iloc[:-1, :, :]
                tmp = self.pn.to_frame(filter_observations=False)
                try:
                    for opt in operations:
                        tmp.eval(opt, inplace=True, **kwargs)
                except:
                    print("some operations is not able to function! ")
                self.pn = tmp.to_panel()
                self.pn[df_name] = target_df

    # def assign(self, axis=None, **kwargs):
    #     """
    #     :param :  默认对一个MultiIndex的level分别为[time, stocks]的DataFrame进行行的操作
    #     可以借鉴的如：axis=1, new_factor_name = lambda x: max(x[factor_name1], x[factor_name2])
    #     ----------------------------------------------------------
    #
    #     """
    #
    #     df_name = self.pn.axes[0][-1]
    #     target_df, self.pn = self.pn.iloc[-1, :, :], self.pn.iloc[:-1, :, :]
    #     tmp = self.pn.to_frame(filter_observations=False)
    #     # ... and then assign
    #     if axis == 1:
    #         for k, v in sorted(kwargs.items()):
    #             tmp[k] = tmp.apply(v, axis=1)
    #     elif axis is None:
    #         results = {}
    #         for k, v in kwargs.items():
    #             results[k] = com._apply_if_callable(v, tmp)
    #         for k, v in sorted(kwargs.items()):
    #             tmp[k] = tmp.apply(v)
    #     # for name, f in zip(new_feature_name, func):
    #     #     tmp[name] = tmp.apply(f, axis=1)
    #     self.pn = tmp.to_panel()
    #     self.pn[df_name] = target_df

    def assign(self, **kwargs):
        """        
        :param :  默认对一个MultiIndex的level分别为[time, stocks]的DataFrame进行行的操作
        可以借鉴的如：axis=1, new_factor_name = lambda x: max(x[factor_name1], x[factor_name2]) 
        ----------------------------------------------------------

        """

        df_name = self.pn.axes[0][-1]
        target_df, self.pn = self.pn.iloc[-1, :, :], self.pn.iloc[:-1, :, :]
        tmp = self.pn.to_frame(filter_observations=False)
        # ... and then assign
        try:

            for k, v in sorted(kwargs.items()):
                tmp[k] = tmp.apply(v, axis=1)
            # for name, f in zip(new_feature_name, func):
            #     tmp[name] = tmp.apply(f, axis=1)
        except:
            print("function not supported!")
        self.pn = tmp.to_panel()
        self.pn[df_name] = target_df





