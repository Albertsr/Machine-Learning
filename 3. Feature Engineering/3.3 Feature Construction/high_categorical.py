# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
import pandas as pd


def high_categorical(df, feature, k=3):
    # df为pandas.DataFrame格式
    # feature为df的某一列高势集离散型特征，为pandas.Series格式
    # k表示上述离散型特征出现频次最高的k个不重复取值
    feature = pd.Series(feature)
    name = feature.name
    feature_val_counts = feature.value_counts()
    
    val_class = list(feature_val_counts[:k].index)
    val_class.append('other')
    feature = feature.apply(lambda val: val if val in val_class else 'other')
    feature_dummies = pd.get_dummies(feature, prefix=name)
    
    df = df.join(feature_dummies)
    df.drop(feature.name, axis=1, inplace=True)
    return df