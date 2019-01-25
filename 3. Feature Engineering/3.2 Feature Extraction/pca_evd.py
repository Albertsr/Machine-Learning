# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from numpy import linalg as LA

class PCA_EVD:
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
         
    # 求标准化矩阵的协方差矩阵
    def matrix_cov(self):
        # rowvar设置为False表示每列代表一个特征，每行代表一个观测值; 默认值为True
        # ddof默认值为1，表示是无偏估计
        cov_matrix = np.cov(self.scale(), rowvar=False, ddof=1)
        return cov_matrix
        
    # 求投影矩阵、特征值、特征向量
    def matrix_eig(self):
        # eigenvectors的每一列即为一个特征向量
        # the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
        eigenvalues, eigenvectors = LA.eig(self.matrix_cov())
        
        # 根据特征值大小对特征值、特征向量降序排列
        eigen_values = eigenvalues[np.argsort(-eigenvalues)]
        eigen_vectors = eigenvectors[np.argsort(-eigenvalues)]
        
        # 选取eigen_vectors的前n_components列，构成的n*n_components型投影矩阵Q
        Q = eigen_vectors[:, :self.n_components]
        return Q, eigen_values, eigen_vectors
    
    # 完成降维
    def pca_result(self):
        Q = self.matrix_eig()[0]
        PCA_result = np.dot(self.scale(), Q)
        assert PCA_result.shape[1] == self.n_components, '降维后矩阵的列数应等于指定的主成分数'
        return PCA_result