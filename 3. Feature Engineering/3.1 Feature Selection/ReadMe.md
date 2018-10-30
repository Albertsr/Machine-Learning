
- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr

---

## 1.Filter (过滤型)

#### 1.1 概述
- Filter方法运用特定的统计指标对单个特征进行评分，评分高的特征优先被选择

---

#### 1.2 优缺点
- **优点**
  - 算法的通用性强，算法复杂度低，适用于大规模数据集
  - 可快速去除大量不相关的特征，适合作为特征的预筛选器

- **缺点**：
  - Filter独立地考察单个特征，不考虑与其他特征之间的联系，被保留的特征可能具有冗余性
  - Filter不考虑特征对模型性能的影响，被保留的特征对于模型性能来说不一定是最优特征
  
---

#### 1.3 常见的Filter方法
- **方差阈**
  - 方差阈是一种无监督方法，通过移除方差低于阈值的特征进行选择
  - sklearn.feature_selection.VarianceThreshold(threshold=0.0)

 
- **卡方检验**
  - 卡方检验只能用于检测**非负特征**与**分类标签列**的独立性，卡方统计量越大，两者越可能相互独立；
  - sklearn.feature_selection.chi2(X, y)


- **互信息**
  - **能捕捉变量之间任何线性或非线性关系：** 既可以用于筛选分类模型的特征，也可以于筛选回归模型的特征
  - **适用于分类模型：** sklearn.feature_selection.mutual_info_classif
  - **适用于回归模型：** sklearn.feature_selection.mutual_info_classif
 

- **F检验**
  - **只能衡量线性关系：** 既可以用于筛选分类模型的特征，也可以于筛选回归模型的特征
  - **适用于分类模型：** 
     - sklearn.feature_selection.f_classif(X, y)
     - Compute the ANOVA F-value for the provided sample
  - **适用于回归模型：**
    - sklearn.feature_selection.f_regression(X, y, center=True)
    - Univariate linear regression tests
---

## 2.Wrapper (封装型)

#### 2.1 概述
- Wrapper根据**外部模型**返回的特征重要性，在迭代过程中递归地剔除不重要的特征

- Wrapper通过**贪心搜索算法，启发式地递归搜索**最佳特征子集，**最佳特征子集**是指所训练的模型具有**最佳的交叉验证性能**

- 外部模型需要具备coef_或feature_importances_属性来对特征重要性进行评估 


#### 2.2 优缺点
- **优点**：能将特征之间的非独立性考虑在内，基于外部模型性能筛选出**独立性与解释能力较强**的特征
- **缺点**：相比其他特征选择方法，有更高的计算代价，筛选出的特征子集更易过拟合

#### 2.3 常见Wrapper方法
- **递归消除特征法(RFE, recursive feature elimination)**
  - 通过**逐步剔除回归系数或重要性较小的特征**来进行特征选择
  - sklearn.feature_selection.RFE(estimator, n_features_to_select=None, step=1, verbose=0) 

- **sequential feature selection algorithms**

- **genetic algorithms**


---

## 3.Embedded(嵌入型)

#### 3.1 概述
- **特征选择过程与模型训练过程融为一体，两者在同一个优化过程中完成，即在模型训练过程中同时进行了特征选择**
- **学习器必须具有衡量特征重要性的属性：** 能返回特征系数coef或者特征重要度(feature importances)的算法才可以做为嵌入法的基学习器，例如线性模型和决策树类算法
- **嵌入型与封装型的区别：** 在模型训练过程中是否具备内生性的特征评价准则

#### 3.2 优缺点
- **优点**：相比wrapper计算消耗更少
- **缺点**：仅限于特定的机器学习算法（specific to a learning machine）


#### 3.3 常见的嵌入型方法
- **L1正则化**：
  - 典型的嵌入式特征选择方法，能有效降低过拟合风险
  - L1正则化可能是不稳定的，如果特征集合中具有共线性特征，则共线性特征可能只保留了一个，没有被选择到的特征不代表不重要
  - 如果要确定哪个特征重要，可再通过L2正则方法交叉检验


- **决策树类模型**：
  - **以CART回归树为例**：若某特征的各个取值作为分裂结点时，平方误差减少量极少，意味着这个特征不能参与CART回归树构建过程，从而在训练CART回归树的同时，也完成了特征选择过程
  - **XGBoost**：既是决策树类模型，同时又带有正则化项
  

#### 3.4 **API**
  - sklearn.feature_selection.SelectFromModel(estimator, threshold=None, prefit=False, norm_order=1)
  - estimator： 模型必须具有feature_importances_ 或者coef_ attribute这里反应特征重要性的属性
  - threshold ：大于等于阈值的特征被保留，默认取'mean'，还可取'median'，以及'1.25*mean'、标量等形式
