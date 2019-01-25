# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
import pandas as pd


def create_time_feature(df, time_feature, start_end_month=3):
    # 确保time_feature为pandas.pd.DatetimeIndex格式
    time_feature = pd.DatetimeIndex(time_feature)
    
    # pandas里面的Timestamp、DatetimeIndex具有is_month_start、is_month_end属性来判断是否为月初月末
    # 参数k用于需要自定义月初月末的范围，即月初k天、月末k天均定义为月初月末
    result_list = []
    for i in time_feature:
        month_len = i.days_in_month  
        month_end = [(month_len - j) for j in range(start_end_month)]
        month_start_end = list(range(1, start_end_month+1))
        month_start_end.extend(month_end)
        result = i.day in month_start_end
        result_list.append(int(result))
    df["Is_Month_Start_End"] = result_list
    
    
    # 与周相关的特征
    # 小写a:返回星期几的英文缩写，如Mon、Tue；大写A则为完整形式
    df["Weekday"] = [i.strftime("%a") for i in time_feature]
    
    # datetime.isoweekday():返回星期索引，取值范围为1～7，以周一作为每周的第一天
    ISO_Weekday = [i.isoweekday() for i in time_feature]
    
    # 根据星期索引判断是否为周末
    df["Is_Weekend"] = [int(i in [6,7]) for i in ISO_Weekday]

    # 大写W：返回在本年度的第几周，周一作为每周的第一天，新年的第一个周一前的日期属于week 0.
    df["Week_Order"] = [i.strftime("%W") for i in time_feature]
    
    
    # 与季节相关的特征
    def Season(x):
        if x in range(1,4):
            return "Spring"
        elif x in range(4,7):
            return "Summer"
        elif x in range(7,10):
             return "Autumn"   
        else:
            return "Winter"
    df["Season"] = time_feature.month.map(Season) 
        
        
    # 与日期、时间相关的特征
    df["Time"] = time_feature.strftime("%H:%M:%S") 
    df["Hour_of_Day"] = [i[:2] for i in df["Time"]]
    
    def Time_Range(x):
        if x >= "06:00:00" and x < "12:00:00":
            return "AM"
        
        elif x >= "12:00:00" and x < "19:00:00":
            return "PM"
        
        elif x >= "19:00:00" and x < "23:00:00":
            return "Night"  
        
        else:
            return "Mid Night"
    df["Time_Range"] = df["Time"].map(Time_Range)
    
    # 小写j: 返回在本年度属于第几天，范围01-366
    # 等价于df["Day_Order"] = [i.dayofyear for i in time_feature]
    df["Day_Order"] = [i.strftime("%j") for i in time_feature]
    
    return df.drop(time_feature.name, axis=1)