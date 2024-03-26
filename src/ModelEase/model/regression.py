import time

import matplotlib.pyplot as plt
import numpy as np

from .. import decorators
from .. import table
from ..dataSet import data_set


class _RegressionModel:
    """
    回归模型的基类
    """
    model = None  # 模型
    name = None  # 模型名称
    model_method = None  # 模型方法

    coef = None  # 系数
    intercept = None  # 截距

    random_state = None  # 随机种子
    data_name = None  # 数据集名称
    x_train = None  # 训练集自变量
    x_test = None  # 测试集自变量
    y_train = None  # 训练集因变量
    y_test = None  # 测试集因变量

    Train_Cost = None  # 训练耗时
    Predict_Cost = None  # 预测耗时

    y_pred = None  # 预测结果
    MSE = None  # 均方误差
    RMSE = None  # 均方根误差
    MAE = None  # 平均绝对误差
    R2 = None  # R2
    adj_R2 = None  # 调整R2
    Explained_Variance = None  # 解释方差

    def __str__(self):
        return f'{self.name} [{self.data_name}]'

    @decorators.cost_record('Init')
    def __init__(self, data: data_set, name: str = None, random_state: int = None):
        if name is None:
            name = self.__class__.__name__ + time.strftime(' %Y-%m-%d %H:%M:%S', time.localtime())
        if name in table:
            raise ValueError('Model name exists. You can use default name or another name.')
        self.name = name
        # 全局变量table注册模型
        table[self.name] = dict()
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = data.random_state
        self.data_name = data.name
        self.x_train = data.x_train
        self.x_test = data.x_test
        self.y_train = data.y_train
        self.y_test = data.y_test

    def __del__(self):
        # 全局变量table销毁模型
        if self.name in table:
            del table[self.name]
            print(f'{self.name} has been deleted from table.')

    # 定义模型
    @decorators.cost_record('Define Model')
    def define_model(self, **kwargs):
        self.model = self.model_method(**kwargs)

    # 训练模型
    @decorators.cost_record('Train', True)
    def train(self):
        self.model.fit(self.x_train, self.y_train)
        self.coef = self.model.coef_
        self.intercept = self.model.intercept_

    # 预测
    @decorators.cost_record('Predict', True)
    def predict(self):
        self.y_pred = self.model.predict(self.x_test)

    # 绘制训练集和测试集的散点图以及模型的拟合直线
    @decorators.cost_record('Scatter')
    def scatter(self):
        plt.scatter(self.y_test, self.y_pred)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=4)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.show()

    # 评估模型
    @decorators.cost_record('Evaluate')
    def evaluate(self):
        self.MSE = np.mean((self.y_pred - self.y_test) ** 2)
        self.RMSE = np.sqrt(self.MSE)
        self.MAE = np.mean(np.abs(self.y_pred - self.y_test))
        self.R2 = self.model.score(self.x_test, self.y_test)
        self.adj_R2 = 1 - (1 - self.R2) * (len(self.y_test) - 1) / (len(self.y_test) - self.x_test.shape[1] - 1)
        self.Explained_Variance = np.var(self.y_pred) / np.var(self.y_test)

        # 全局变量table更新模型评估结果
        table[self.name].update({
            'MSE': self.MSE,
            'RMSE': self.RMSE,
            'MAE': self.MAE,
            'R2': self.R2,
            'adj_R2': self.adj_R2,
            'Explained_Variance': self.Explained_Variance
        })

        # 输出评估结果
        print(f'{self.name} [{self.data_name}]')
        print(f'MSE: {self.MSE}')
        print(f'RMSE: {self.RMSE}')
        print(f'MAE: {self.MAE}')
        print(f'R2: {self.R2}')
        print(f'adj_R2: {self.adj_R2}')
        print(f'Explained_Variance: {self.Explained_Variance}')

    def auto(self, **kwargs):
        self.define_model(**kwargs)
        self.train()
        self.predict()
        self.scatter()
        print('coef: \n', self.coef)
        print('intercept: ', self.intercept)
        self.evaluate()


# Linear Regression
class LinearRegression(_RegressionModel):
    from sklearn.linear_model import LinearRegression
    model_method = LinearRegression


# Ridge Regression
class RidgeRegression(_RegressionModel):
    from sklearn.linear_model import Ridge
    model_method = Ridge

    def define_model(self, alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto',
                     positive=False, random_state=None):
        self.model = self.model_method(alpha=alpha, fit_intercept=fit_intercept, copy_X=copy_X, max_iter=max_iter,
                                       tol=tol, solver=solver, positive=positive, random_state=random_state)
