# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from numpy import linalg as LA

class PCA_SVD:
    # 参数n_components为保留的主成分数
    def __init__(self, matrix, n_components=None):
        self.matrix = matrix
        self.n_components = matrix.shape[1] if n_components==None else n_components
    
    # 自定义标准化方法
    def scale(self):
        def scale_vector(vector):
            delta = vector - np.mean(vector)
            std = np.std(vector, ddof=0)
            return delta / std
        matrix_scaled = np.apply_along_axis(arr=self.matrix, func1d=scale_vector, axis=0)
        return matrix_scaled
     
    # 对标准化后的矩阵进行奇异值分解    
    def matrix_svd(self):
        # 令A为m*n型矩阵，则U、V分别为m阶、n阶正交矩阵
        # U的每一个列向量都是A*A.T的特征向量，也称为左奇异向量
        # V的每一个行向量都是A.T*A的特征向量，也称为右奇异向量
        # sigma是由k个降序排列的奇异值构成的向量，其中k = min(matrix.shape)
        U, sigma, V =  LA.svd(self.scale()) 
        
        # 非零奇异值的个数不会超过原矩阵的秩，从而不会超过矩阵维度的最小值
        assert len(sigma) == min(self.matrix.shape)
        return U, sigma, V 
    
    # 通过矩阵V进行PCA，返回最终降维后的矩阵
    def pca_result(self):
        sigma, V = self.matrix_svd()[1], self.matrix_svd()[2]
        # Q为投影矩阵，由V的前n_components个行向量转置后得到
        Q = V[:self.n_components, :].T
        # 计算标准化后的矩阵在Q上的投影，得到PCA的结果
        matrix_pca = np.dot(self.scale(), Q)
        # matrix_pca的列数应等于保留的主成分数
        assert matrix_pca.shape[1] == self.n_components
        return matrix_pca