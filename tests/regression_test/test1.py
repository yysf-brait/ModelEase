"""
测试回归模型
测试auto()方法
"""

import os

import pandas as pd

import src.ModelEase as me  # noqa

me.n_jobs = 16

# 读入california数据集
# 如果当前目录下有california.csv文件，读入数据集
# 如果没有，路径切换到tests/classification_test，读入数据集
if not os.path.exists('california.csv'):
    os.chdir('tests//regression_test')
df = pd.read_csv('california.csv').iloc[:500, :]

# 创建数据集
data = me.data_set(df, x_index=[0, 1, 2, 3, 4, 5, 6, 7], y_index=8, test_size=0.2, name='California Housing')

# Linear Regression
Linear = me.model.regression.LinearRegression(data, name='Linear Regression')
Linear.auto()

# Ridge Regression
Ridge = me.model.regression.RidgeRegression(data, name='Ridge Regression')
Ridge.auto()

# Decision Tree Regression
DecisionTree = me.model.regression.DecisionTreeRegression(data, name='Decision Tree Regression')
DecisionTree.auto()

# SVM Regression
SVR = me.model.regression.SVR(data, name='SVM Regression')
SVR.auto(C=0.2, cache_size=4*1024, kernel='linear')

# KNN Regression
KNN = me.model.regression.KNN(data, name='KNN Regression')
KNN.auto()

print(me.comparison())
