- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr
---

### 数据探索实例(请点击以下链接)
##### [1.对Titanic数据集进行探索](https://nbviewer.jupyter.org/github/Albertsr/Machine-Learning/blob/master/1.%20Data%20Exploration/%E4%B8%93%E9%A2%981%EF%BC%9A%E5%88%86%E7%B1%BB%E9%97%AE%E9%A2%98%E7%9A%84%E6%95%B0%E6%8D%AE%E6%8E%A2%E7%B4%A2%28%E4%BB%A5Titanic%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%BA%E4%BE%8B%29.ipynb)

##### [2.回归问题中的相关系数矩阵与热力图(以Boston数据集为例)](https://nbviewer.jupyter.org/github/Albertsr/Machine-Learning/blob/master/1.%20Data%20Exploration/%E4%B8%93%E9%A2%982%EF%BC%9A%E5%9B%9E%E5%BD%92%E9%97%AE%E9%A2%98%E4%B8%AD%E7%9A%84%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0%E7%9F%A9%E9%98%B5%E4%B8%8E%E7%83%AD%E5%8A%9B%E5%9B%BE%28%E4%BB%A5Boston%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%BA%E4%BE%8B%29.ipynb)

##### [3.对iris数据集进行数据探索](https://nbviewer.jupyter.org/github/Albertsr/Machine-Learning/blob/master/1.%20Data%20Exploration/%E4%B8%93%E9%A2%983%EF%BC%9A%E5%AF%B9iris%E6%95%B0%E6%8D%AE%E9%9B%86%E8%BF%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E6%8E%A2%E7%B4%A2.ipynb)

---

### 数据探索的基本思路（以Titanic数据集为例）

#### 1.1 离散型特征的探索
- 各等级舱位等级Pclass的人数
  - train["Pclass"].value_counts()
  - sns.countplot(x='Pclass', data=train)

- 船舱等级对生还率的影响
  - train.pivot_table(index=["Pclass"], values=["Survived"], aggfunc="mean")
  - sns.barplot(x='Pclass', y='Survived', data=train)

- 与其他离散型变量的联合探索
  -  对Pclass、Sex联合探索：无论哪个船舱等级，女性都有更高的生还率
     - train.pivot_table(index=["Pclass"], values=["Survived"], columns=["Sex"])
     - sns.barplot(x='Pclass', y='Survived', data=train, hue='Sex', dodge=True)

  - 对Pclass、Embarked进行联合探索：S港口登录的乘客生还率最低，C港口登录的乘客生还率稍高于Q
    - train.pivot_table(index=["Pclass"], values=["Survived"], columns=["Embarked"])
    - sns.barplot(x='Pclass', y='Survived', data=train, hue='Embarked', dodge=True) 
  
---

#### 1.2 连续型特征的探索

- 通过箱型图发现Fare特征存在一些极端的异常值，对这些特征修改为95分位点的值
  - sns.boxplot(train["Fare"], ax=axes[0])  
  - Fare_per_95 = np.percentile(train["Fare"], 95)
  - train["Fare"][train["Fare"] >= Fare_per_95] = Fare_per_95

- 通过连续型变量的KDE曲线，探索连续型变量对分类结果的影响，还可以为连续型特征的离散化处理提供参考
  - 查看生还与否的年龄Age密度
    - sns.distplot(train['Age'][train['Survived']==0], ax=axes[0])
    - sns.distplot(train['Age'][train['Survived']==1],ax=axes[0])
   
  - 查看生还与否的费用Fare密度
    - sns.distplot(train['Fare'][train['Survived']==0], ax=axes[1])
    - sns.distplot(train['Fare'][train['Survived']==1], ax=axes[1])

- 某些特征虽然是连续型变量，但是取值较少，分布不均匀，可作为离散型变量处理
   - 例如家属个数
   - sns.barplot(x="Parch", y="Survived", data=train)
   - sns.barplot(x="SibSp", y="Survived", data=train)
   
---

#### 3. 离散型与连续性特征的联合探索

- 无论什么性别，舱位与年龄成反比；无论哪个舱位，男性要年长于女性
    - train.pivot_table(index=["Pclass"], values=["Age", 'Fare'], columns=["Sex"])
    - sns.barplot(x='Pclass', y='Age', data=train, hue="Sex", dodge=True, errwidth=2)
    
- 舱位与票价成正比；无论哪个舱位，女性支付的费用都高于男性
    - train.pivot_table(index=["Pclass"], values=["Fare"], columns=["Sex"])
    - sns.barplot(x='Pclass', y='Fare', data=train, hue="Sex", dodge=True, errwidth=2) 
