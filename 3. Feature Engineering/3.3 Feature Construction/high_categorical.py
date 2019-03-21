# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
import pandas as pd


def high_categorical(dataframe, high_discrete, k=3):
    # dataframe为pandas.DataFrame格式
    # high_discrete为dataframe的某一列高势集离散型特征，为pandas.Series格式
    # k表示上述离散型特征出现频次最高的k个不重复取值
    
    value_counts = high_discrete.value_counts()
    top_categories = list(value_counts[:k].index)
    top_categories.append('other')
    
    high_discrete = high_discrete.apply(lambda category: category if category in top_categories else 'other')
    #print(high_discrete)
    feature_dummies = pd.get_dummies(high_discrete, prefix=high_discrete.name)
    
    dataframe = dataframe.join(feature_dummies)
    dataframe.drop(high_discrete.name, axis=1, inplace=True)
    return dataframe
