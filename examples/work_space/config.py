# encoding:utf-8

from fxdayu_data.data_api import *
from fxdayu_data.handler.mongo_handler import MongoHandler

# 先实例化一个IO类对象(MongoDB)
handler = MongoHandler.params(host="192.168.0.101", port=27017)
# 实例化K线配置
candle = Candle()
# 建立IO对象，K线周期和数据库之间的索引关系
candle.set(handler, M1='stock_1min', H1='stock_h', D='tushare_D')
# 建立复权因子的索引关系
candle.set_adjust(handler, 'adjust', 'after')
# 实例化基本面数据配置
factor = Factor()
# 建立基本面数据与数据库之间的索引关系
factor.set(handler, 'fundamental')