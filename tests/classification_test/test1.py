"""
测试分类模型
测试auto(search_best_params=False)方法
"""

import os

import pandas as pd

import src.ModelEase as me  # noqa

# 读入iris数据集
# 如果当前目录下有iris.csv文件，读入数据集
# 如果没有，路径切换到tests/classification_test，读入数据集
if not os.path.exists('iris.csv'):
    os.chdir('tests//classification_test')
df = pd.read_csv('iris.csv')

# 创建数据集
data = me.data_set(df, x_index=[0, 1, 2, 3], y_index=4, test_size=0.2, name='Iris')

# DecisionTree
DecisionTree = me.model.classification.DecisionTree(data, name='DecisionTree')
DecisionTree.auto()

# CNBayes
CNBayes = me.model.classification.CNBayes(data, name='CNBayes')
CNBayes.auto()

# GNBayes
GNBayes = me.model.classification.GNBayes(data, name='GNBayes')
GNBayes.auto()

# KNN
KNN = me.model.classification.KNN(data, name='KNN')
KNN.auto()

# RandomForest
RandomForest = me.model.classification.RandomForest(data, name='RandomForest')
RandomForest.auto()

# AdaBoost
AdaBoost = me.model.classification.AdaBoost(data, name='AdaBoost')
AdaBoost.auto()

# SVM
SVM = me.model.classification.SVM(data, name='SVM')
SVM.auto()

# LogisticRegression
LogisticRegression = me.model.classification.LogisticRegression(data, name='LogisticRegression')
LogisticRegression.auto()

print(me.comparison())
