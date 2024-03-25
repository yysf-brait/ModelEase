import matplotlib.pyplot as plt
import numpy as np

from .. import decorators
from .. import model_list
from ..dataSet import data_set


class _RegressionModel:
    """
    模型的基类
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

    train_cost = None  # 训练耗时
    predict_cost = None  # 预测耗时

    y_pred = None  # 预测结果
    MSE = None  # 均方误差
    RMSE = None  # 均方根误差
    MAE = None  # 平均绝对误差
    R2 = None  # R2
    adj_R2 = None  # 调整R2
    Explained_Variance = None  # 解释方差

    def __str__(self):
        return f'{self.name} [{self.data_name}]'

    def __init__(self, data: data_set, name: str = 'Model', random_state: int = None):
        self.name = name
        # 全局变量model_list注册模型
        model_list[self.name] = dict()
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
        # 全局变量model_list销毁模型
        if self.name in model_list:
            del model_list[self.name]

    # 定义模型
    def define_model(self, **kwargs):
        self.model = self.model_method(**kwargs)

    # 训练模型
    def train(self):
        self.model.fit(self.x_train, self.y_train)
        self.coef = self.model.coef_
        self.intercept = self.model.intercept_

    # 预测
    def predict(self):
        self.y_pred = self.model.predict(self.x_test)

    # 绘制训练集和测试集的散点图以及模型的拟合直线
    def scatter(self):
        plt.scatter(self.y_test, self.y_pred)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=4)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.show()

    # 评估模型
    def evaluate(self):
        self.MSE = np.mean((self.y_pred - self.y_test) ** 2)
        self.RMSE = np.sqrt(self.MSE)
        self.MAE = np.mean(np.abs(self.y_pred - self.y_test))
        self.R2 = self.model.score(self.x_test, self.y_test)
        self.adj_R2 = 1 - (1 - self.R2) * (len(self.y_test) - 1) / (len(self.y_test) - self.x_test.shape[1] - 1)
        self.Explained_Variance = np.var(self.y_pred) / np.var(self.y_test)

        # 全局变量model_list注册模型
        model_list[self.name] = dict(
            MSE=self.MSE,
            RMSE=self.RMSE,
            MAE=self.MAE,
            R2=self.R2,
            adj_R2=self.adj_R2,
            Explained_Variance=self.Explained_Variance
        )

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
        self.evaluate()
        self.scatter()


# Linear Regression
class LinearRegression(_RegressionModel):
    from sklearn.linear_model import LinearRegression
    model_method = LinearRegression

    # 定义构造函数
    @decorators.cost_record('Class[LinearRegression] Init')
    def __init__(self, data: data_set, name: str = 'LinearRegression', random_state: int = None):
        super().__init__(data, name, random_state)

    # 定义模型
    @decorators.cost_record('Class[LinearRegression] Define Model')
    def define_model(self, **kwargs):
        super().define_model(**kwargs)

    # 训练模型
    @decorators.cost_record('Class[LinearRegression] Train')
    def train(self):
        super().train()

    # 预测
    @decorators.cost_record('Class[LinearRegression] Predict')
    def predict(self):
        super().predict()

    @decorators.cost_record('Class[LinearRegression] Scatter')
    def scatter(self):
        super().scatter()

    # 评估模型
    @decorators.cost_record('Class[LinearRegression] Evaluate')
    def evaluate(self):
        super().evaluate()
