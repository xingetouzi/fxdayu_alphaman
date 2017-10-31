from __future__ import division

import os
from functools import partial
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from .preprocessing import Basic

__all__ = [
    'RollingClf',
    'RollingRgr',
    'StaticClf',
    'StaticRgr'
    ]


def zero_cut(array):
    array = np.where(array > 0, 1, -1)
    return array


def pct_cut(array, pct_list, labels=None):
    if labels is None:
        labels = list(range(1, len(pct_list) + 2))
    assert len(pct_list) + 1 == len(labels)

    def base_cut(x, y, z):
        zz = z.pop()
        if len(y) != 0:
            yy = y.pop()
            return np.where(x > yy, zz, base_cut(x, y, z))
        if len(y) == 0:
            return zz

    array_pct = np.percentile(array, pct_list, axis=0)

    if array_pct.size == array_pct.shape[0]:
        array = base_cut(array, list(array_pct), list(labels))
    else:
        for j in range(array_pct.shape[1]):
            array[:, j] = base_cut(array[:, j], list(array_pct[:, j]), list(labels))

    return array


def equal_cut(array, num, labels=None):
    if labels is None:
        labels = list(range(1, num + 1))
    assert num == len(labels)
    pct_list = [(i + 1) / num * 100 for i in range(num - 1)]
    return pct_cut(array, pct_list, labels=labels)


class _Classifier(object):
    def __init__(self, PN):
        self.pn = PN
        self.cut_methods = {
            "zero": zero_cut,
            "pct": pct_cut,
            "equal": equal_cut
        }
        self.classifiers = {
            'ABC': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=5)

        }

    def cut(self, func, specific=None, **kwargs):
        """
        用于对特征值进行横截面上的切割，因为像决策树的分类方法对连续数值很容易过拟合；
        同时，如果要以横截面打分的方式来处理特征也可以使用这个方法
        :param func: : "pct", "equal",  or callable function
        :param specific: str，特征名，指定特定的特征列来切割
        :param kwargs: "pct": 参数有二，pct_list和labels，依照哪几个分位数来切，长度大于labels，
                                如pct_list=[30,70], labels=[1,2,3]或None;
                        "equal":参数有二，num和labels，前者分割成几段，长度等于labels，或labels等于None;
        :return: 
        """
        if func == 'pct':
            func = partial(pct_cut, **kwargs)
        elif func == 'equal':
            func = partial(equal_cut, **kwargs)
        elif not callable(func):
            raise ValueError

        df_name = self.pn.axes[0][-1]
        tmp_df, self.pn = self.pn.iloc[-1, :, :], self.pn.iloc[:-1, :, :]

        if specific is None:

            self.pn = self.pn.to_frame().groupby(level=0).transform(func).to_panel()

        else:

            tmp = self.pn.to_frame()
            tmp[:, specific] = tmp[:, specific].groupby(level=0).transform(func)
            self.pn = tmp.to_panel()

        self.pn[df_name] = tmp_df


class _Base(Basic):
    def __init__(self, PN):
        """

        :param PN: the axes of input panel are [factors, time, stocks] 
                    and the last column of factor should be 'ret';
        """
        super(_Base, self).__init__(PN)
        self.models = []
        self.result = None

    def save_models(self, path=None):
        if path is None:
            path = os.path.join(os.getcwd(), "temp")
        print(path)
        os.makedirs(path)
        joblib.dump(self.models, os.path.join(path, "models.pkl"))


class _BaseRolling(_Base):
    def __init__(self, PN):
        """
        
        :param PN: the axes of input panel are [factors, time, stocks] 
                    and the last column of factor should be 'ret';
        """
        super(_BaseRolling, self).__init__(PN)
        self.start = None
        self.forward = None
        self.look_back = None
        self.end = None
        self.freq = None
        self.trainX_iterator = None
        self.predictX_iterator = None
        self.trainY_iterator = None
        self.predictY_iterator = None

    def split(self, ret_fwd, train_len, remodel_freq, start=None, end=None):  # 要改
        """
        split： dropna会在这里进行
        :param ret_fwd: 收益率计算的期数
        :param train_len: 训练的长度
        :param remodel_freq: 重新建模的期数，也是每一次预测的期数
        :param start: int，预测的第一天，划分最初的训练集和预测集
        :param end: int，结束预测的一天
        """
        self.forward = ret_fwd
        self.look_back = train_len + ret_fwd - 1
        self.freq = remodel_freq
        if start is None:
            self.start = self.look_back
        else:
            self.start = start
        if end is None:
            self.end = self.pn.shape[1] - ret_fwd
        else:
            self.end = end - self.freq
        self.trainX_iterator = (
            self.pn.iloc[:, i-self.look_back:i - ret_fwd + 1, :].to_frame().iloc[:, :-1]
            for i in range(self.start, self.end, self.freq)
        )

        self.trainY_iterator = (
            self.pn.iloc[:, i-self.look_back:i - ret_fwd + 1, :].to_frame().iloc[:, -1:]
            for i in range(self.start, self.end, self.freq)
        )

        self.predictX_iterator = (
            self.pn.iloc[:, i:i+self.freq, :].to_frame().iloc[:, :-1].values
            for i in range(self.start, self.end, self.freq)
        )

        self.predictY_iterator = (
            self.pn.iloc[:, i:i+self.freq, :].to_frame().iloc[:, -1:]
            for i in range(self.start, self.end, self.freq)
        )

    def predict(self):
        pass


class RollingClf(_Classifier, _BaseRolling):

    def __init__(self, PN):
        _Classifier.__init__(self, PN)
        _BaseRolling.__init__(self, PN)
        self.label_iterator = None

    def label(self, func, xsection=True, **kwargs):

        """
            在横截面上设定分类器训练目标
           :param func: : "pct", "equal",  or callable function
           :param xsection: True or False, True则是在每个横截面上做标签， False则是各个切割出来的训练集为整体做标签
           :param kwargs: "pct": 参数有二，pct_list和labels，依照哪几个分位数来切，长度大于labels，
                                   如pct_list=[30,70], labels=[1,2,3]或None;
                           "equal":参数有二，num和labels，前者分割成几段，长度等于labels，或labels等于None;
                           另： 以上两个方法都包含参数labels， 若将labels中的元素设为np.nan则该分类的样本不会进入训练集
                           "zero":无参数，直接按正负切
           :return: 
        """

        if func == 'pct':
            func = partial(pct_cut, **kwargs)

        elif func == 'zero':
            func = zero_cut

        elif func == 'equal':
            func = partial(equal_cut, **kwargs)

        elif not callable(func):
            raise ValueError

        if xsection:
            tmp_labels = self.pn.iloc[-1:, :, :].to_frame().groupby(level=0).transform(func)
            self.label_iterator = (
                tmp_labels.reindex_like(it).dropna() for it in self.trainY_iterator
            )
            return pd.concat([self.pn.iloc[-1:, :, :].to_frame(), tmp_labels.rename(columns={'ret': 'label'})], axis=1)
        else:
            self.label_iterator = (
                it.transform(func).dropna() for it in self.trainY_iterator
            )

    def predict(self, classifier='ABC', save=False):
        """
        
        :param classifier: str or func, 分类器
        :param save: 是否储存分类器
        :return: 
        """
        if classifier in self.classifiers:
            pre_model = self.classifiers[classifier]
        elif hasattr(classifier, "fit"):
            pre_model = classifier
        else:
            raise ValueError("wrong classifier input")

        if save & (len(self.models) != 0):
            self.models = []
        result = []

        for train_X, label, predict_X, true_y in zip(self.trainX_iterator, self.label_iterator,
                                                     self.predictX_iterator, self.predictY_iterator):
            if train_X.size == 0 or predict_X.size == 0:
                print("found empty panel/array")
                continue
            else:
                ind = true_y.index
                model = clone(pre_model)
                model = model.fit(train_X.reindex(index=label.index), label.values.ravel())
                if save:
                    self.models.append(model)
                prediction = model.predict(predict_X).reshape(-1, 1)
                proba = model.predict_proba(predict_X)
                prediction = pd.DataFrame(np.append(prediction, proba, axis=1),
                                          index=ind, columns=['predict']+model.classes_.tolist())
                result_slice = pd.concat([true_y, prediction], axis=1)
                result.append(result_slice)

        self.result = pd.concat(result, axis=0)


class RollingRgr(_BaseRolling):

    def __init__(self, PN):
        super(RollingRgr, self).__init__(PN)
        self.regressors = {
            'Linear': LinearRegression(),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'E-N': ElasticNet()
        }

    def predict(self, regressor="Ridge", save=False):
        if regressor in self.regressors:
            pre_model = self.regressors[regressor]
        elif hasattr(regressor, "fit"):
            pre_model = regressor
        else:
            raise ValueError("wrong regressors given")
        if save & (len(self.models) != 0):
            self.models = []

        result = []
        for train_X, train_y, predict_X, true_y in zip(self.trainX_iterator, self.trainY_iterator,
                                                       self.predictX_iterator, self.predictY_iterator):
            if train_X.size == 0 or predict_X.size == 0:
                print("found empty panel/array")
                continue
            else:
                ind = true_y.index
                model = clone(pre_model)
                model = model.fit(train_X, train_y.values.ravel())
                self.models.append(model)
                prediction = model.predict(predict_X)
                prediction = pd.Series(prediction, index=ind, name="predict")
                result_slice = pd.concat([true_y, prediction], axis=1)
                result.append(result_slice)

        self.result = pd.concat(result, axis=0)


class _BaseStatic(_Base):

    def __init__(self, PN):
        """
        :param PN: the axes of input panel are [factors, time, stocks] 
                    and the last column of factor should be 'ret';
        """
        super(_BaseStatic, self).__init__(PN)
        self.start = None
        self.forward = None
        self.look_back = None
        self.end = None
        self.trainX = None
        self.predictX = None
        self.trainY = None
        self.predictY = None

    def split(self, ret_fwd, start, train_len=None, end=None):
        """
        the axes of panel passed should be [factors, time, stocks],
                    and the last two columns of factor should be 'ret';
        :param ret_fwd: the forward? periods by which return is calculated 
        :param train_len: int or string, int means the number of samples prepared for training;
                        string means loc of first training sample 
        :param start: int or string, locate the first predicting instance 
        :param end: int or string, locate the last predicting instance
        """
        self.forward = ret_fwd
        if isinstance(start, str):
            self.start = self.pn.axes[1].get_loc(start)
            if isinstance(self.start, slice):
                self.start = self.start.start
        elif isinstance(start, int):
            self.start = start
        if isinstance(train_len, str):
            train_start = self.pn.axes[1].get_loc(train_len)
            if isinstance(train_start, slice):
                train_start = train_start.start
            self.look_back = self.start - train_start - ret_fwd + 1
        elif train_len is None:
            self.look_back = self.start
        elif isinstance(train_len, int):
            self.look_back = train_len + ret_fwd - 1
        if isinstance(end, str):
            self.end = self.pn.axes[1].get_loc(end)
            if isinstance(self.end, slice):
                self.end = self.end.start
        elif isinstance(end, int):
            self.end = end
        elif end is None:
            self.end = self.pn.shape[1]

        # print(self.start, self.look_back, self.end)
        self.trainX = self.pn.iloc[:, self.start - self.look_back: self.start - ret_fwd + 1, :]\
                          .to_frame().iloc[:, :-1]
        self.trainY = self.pn.iloc[:, self.start - self.look_back: self.start - ret_fwd + 1, :]\
                          .to_frame().iloc[:, -1:]
        self.predictX = self.pn.iloc[:, self.start: self.end, :].to_frame().iloc[:, :-1].values
        self.predictY = self.pn.iloc[:, self.start: self.end, :].to_frame().iloc[:, -1:]

    def predict(self):
        pass


class StaticClf(_Classifier, _BaseStatic):

    def __init__(self, PN):
        _Classifier.__init__(self, PN)
        _BaseStatic.__init__(self, PN)
        self.labeled = None

    def label(self, func, xsection=True, **kwargs):
        """
            设定分类器训练目标
           :param func: : "pct", "equal",  or callable function
           :param xsection: True or False， True则是在每个横截面上做标签， False则是在切割出来的训练集上做标签
           :param kwargs: "pct": 参数有二，pct_list和labels，依照哪几个分位数来切，长度大于labels，
                                   如pct_list=[30,70], labels=[1,2,3]或None;
                           "equal":参数有二，num和labels，前者分割成几段，长度等于labels，或labels等于None;
                           另： 以上两个方法都包含参数labels， 若将labels中的元素设为np.nan则该分类的样本不会进入训练集
                           "zero":无参数，直接按正负切
           :return: 
        """

        if func == 'pct':
            func = partial(pct_cut, **kwargs)

        elif func == 'zero':
            func = zero_cut

        elif func == 'equal':
            func = partial(equal_cut, **kwargs)

        elif not callable(func):
            raise ValueError

        if xsection:
            self.labeled = self.trainY.groupby(level=0).transform(func).dropna()

        else:
            self.labeled = self.trainY.transform(func).dropna()

    def predict(self, classifier='ABC', save=False):

        if classifier in self.classifiers:
            pre_model = self.classifiers[classifier]
        elif hasattr(classifier, "fit"):
            pre_model = classifier
        else:
            raise ValueError("wrong classifier input!")

        if save & (len(self.models) != 0):
            self.models = []

        if self.trainX.size == 0 or self.predictX.size == 0:
            print("found empty panel/array")
        else:
            ind = self.predictY.index
            model = clone(pre_model)
            model = model.fit(self.trainX.reindex(index=self.labeled.index), self.labeled.values.ravel())
            if save:
                self.models.append(model)
            prediction = model.predict(self.predictX).reshape(-1, 1)
            proba = model.predict_proba(self.predictX)
            prediction = pd.DataFrame(np.append(prediction, proba, axis=1),
                                      index=ind, columns=['predict']+model.classes_.tolist())
            result = pd.concat([self.predictY, prediction], axis=1)
            self.result = result


class StaticRgr(_BaseStatic):

    def __init__(self, PN):
        super(StaticRgr, self).__init__(PN)
        self.regressors = {
            'Linear': LinearRegression(),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'E-N': ElasticNet()
        }

    def predict(self, regressor="Ridge", save=False):
        if regressor in self.regressors:
            pre_model = self.regressors[regressor]
        elif hasattr(regressor, "fit"):
            pre_model = regressor
        else:
            raise ValueError("wrong regressors given")

        if save & (len(self.models) != 0):
            self.models = []

        if self.trainX.size == 0 or self.predictX.size == 0:
            print("found empty panel/array")
        else:
            ind = self.predictY.index
            model = clone(pre_model)
            model = model.fit(self.trainX, self.trainY.values.ravel())
            if save:
                self.models.append(model)
            prediction = model.predict(self.predictX)
            prediction = pd.Series(prediction, index=ind, name="predict")
            result = pd.concat([self.predictY, prediction], axis=1)
            self.result = result


