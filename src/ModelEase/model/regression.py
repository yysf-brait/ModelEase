import time

import matplotlib.pyplot as plt
import numpy as np

from .. import decorators
from .. import n_jobs
from .. import table
from ..dataSet import data_set


class _RegressionModel:
    """
    回归模型的基类
    """
    model = None  # 模型
    name = None  # 模型名称
    model_method = None  # 模型方法

    feature_contribution = None  # 特征贡献度

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
    MAPE = None  # 平均绝对百分比误差
    SMAPE = None  # 对称平均绝对百分比误差
    RMSPE = None  # 根均方百分比误差
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

    # 获取特征贡献度
    def record_feature_contribution(self):
        pass

    # 训练模型
    @decorators.cost_record('Train', True)
    def train(self):
        self.model.fit(self.x_train, self.y_train)

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
        self.record_feature_contribution()
        if self.feature_contribution is not None:
            print(f'feature_contribution: {self.feature_contribution}')
        self.MSE = np.mean((self.y_pred - self.y_test) ** 2)
        self.RMSE = np.sqrt(self.MSE)
        self.MAE = np.mean(np.abs(self.y_pred - self.y_test))
        self.MAPE = np.mean(np.abs((self.y_pred - self.y_test) / self.y_test))
        self.SMAPE = 2 * np.mean(np.abs(self.y_pred - self.y_test) / (np.abs(self.y_pred) + np.abs(self.y_test)))
        self.RMSPE = np.sqrt(np.mean(((self.y_pred - self.y_test) / self.y_test) ** 2))
        self.R2 = self.model.score(self.x_test, self.y_test)
        self.adj_R2 = 1 - (1 - self.R2) * (len(self.y_test) - 1) / (len(self.y_test) - self.x_test.shape[1] - 1)
        self.Explained_Variance = np.var(self.y_pred) / np.var(self.y_test)

        # 全局变量table更新模型评估结果
        table[self.name].update({
            'MSE': self.MSE,
            'RMSE': self.RMSE,
            'MAE': self.MAE,
            'MAPE': self.MAPE,
            'SMAPE': self.SMAPE,
            'RMSPE': self.RMSPE,
            'R2': self.R2,
            'adj_R2': self.adj_R2,
            'Explained_Variance': self.Explained_Variance
        })

        # 输出评估结果
        print(f'{self.name} [{self.data_name}]')
        print(f'MSE: {self.MSE}')
        print(f'RMSE: {self.RMSE}')
        print(f'MAE: {self.MAE}')
        print(f'MAPE: {self.MAPE}')
        print(f'SMAPE: {self.SMAPE}')
        print(f'RMSPE: {self.RMSPE}')
        print(f'R2: {self.R2}')
        print(f'adj_R2: {self.adj_R2}')
        print(f'Explained_Variance: {self.Explained_Variance}')

    # 模型应用于给定数据集
    def apply(self, x):
        return self.model.predict(x)

    def auto(self, **kwargs):
        self.define_model(**kwargs)
        self.train()
        self.predict()
        self.scatter()
        self.evaluate()


# Linear Regression
class LinearRegression(_RegressionModel):
    from sklearn.linear_model import LinearRegression
    model_method = LinearRegression

    def record_feature_contribution(self):
        self.feature_contribution = dict()
        self.feature_contribution['coef'] = self.model.coef_
        self.feature_contribution['intercept'] = self.model.intercept_


# Ridge Regression
class RidgeRegression(_RegressionModel):
    from sklearn.linear_model import Ridge
    model_method = Ridge

    def define_model(self, alpha=1.0, *, fit_intercept=True, max_iter=None, tol=0.0001, solver='auto',
                     positive=False, random_state=None):
        self.model = self.model_method(alpha=alpha, fit_intercept=fit_intercept, copy_X=True, max_iter=max_iter,
                                       tol=tol, solver=solver, positive=positive, random_state=random_state)

    def record_feature_contribution(self):
        self.feature_contribution = dict()
        self.feature_contribution['coef'] = self.model.coef_
        self.feature_contribution['intercept'] = self.model.intercept_


# Decision Tree Regression
class DecisionTreeRegression(_RegressionModel):
    from sklearn.tree import DecisionTreeRegressor
    model_method = DecisionTreeRegressor

    def define_model(self, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2,
                     min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                     max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0, monotonic_cst=None):
        """
        criterion: {"squared_error", "friedman_mse", "absolute_error", "poisson"}, default="squared_error"
        splitter: {"best", "random"}, default="best"
        max_depth: int, default=None
        min_samples_split: int or float, default=2
        min_samples_leaf: int or float, default=1
        min_weight_fraction_leaf: float, default=0.0
        max_features: int, float or {"sqrt", "log2"}, default=None
        random_state: int, RandomState instance or None, default=model.random_state
        max_leaf_nodes: int, default=None
        min_impurity_decrease: float, default=0.0
        ccp_alpha: non-negative float, default=0.0
        monotonic_cst: array-like of int of shape (n_features), default=None
        """
        self.model = self.model_method(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                       random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                       min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha,
                                       monotonic_cst=monotonic_cst)

    def record_feature_contribution(self):
        self.feature_contribution = self.model.feature_importances_


# Random Forest Regression
class RandomForestRegression(_RegressionModel):
    from sklearn.ensemble import RandomForestRegressor
    model_method = RandomForestRegressor

    def define_model(self, n_estimators=100, *, criterion='squared_error', max_depth=None, min_samples_split=2,
                     min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None,
                     min_impurity_decrease=0.0, bootstrap=True, oob_score=False, random_state=None,
                     verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None):
        """
        n_estimators: int, default=100
        criterion: {"squared_error", "friedman_mse", "absolute_error", "poisson"}, default="squared_error"
        max_depth: int, default=None
        min_samples_split: int or float, default=2
        min_samples_leaf: int or float, default=1
        min_weight_fraction_leaf: float, default=0.0
        max_features: int, float or {"sqrt", "log2"}, default=1.0
        max_leaf_nodes: int, default=None
        min_impurity_decrease: float, default=0.0
        bootstrap: bool, default=True
        oob_score: bool, default=False
        random_state: int, RandomState instance or None, default=model.random_state
        verbose: int, default=0
        warm_start: bool, default=False
        ccp_alpha: non-negative float, default=0.0
        max_samples: int or float, default=None
        monotonic_cst: array-like of int of shape (n_features), default=None
        """
        self.model = self.model_method(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                       max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                       bootstrap=bootstrap, oob_score=oob_score, random_state=random_state,
                                       verbose=verbose,
                                       warm_start=warm_start, ccp_alpha=ccp_alpha, max_samples=max_samples,
                                       monotonic_cst=monotonic_cst)

    def record_feature_contribution(self):
        self.feature_contribution = self.model.feature_importances_


# AdaBoost Regression
class AdaBoostRegression(_RegressionModel):
    from sklearn.ensemble import AdaBoostRegressor
    model_method = AdaBoostRegressor

    def define_model(self, estimator=None, *, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None):
        """
        estimator: object, default=None
        n_estimators: int, default=50
        learning_rate: float, default=1.0
        loss{‘linear’, ‘square’, ‘exponential’}, default=’linear’
        random_state: int, RandomState instance or None, default=model.random_state
        """
        self.model = self.model_method(estimator=estimator, n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       loss=loss, random_state=random_state)

    def record_feature_contribution(self):
        self.feature_contribution = self.model.feature_importances_


# SVR
class SVR(_RegressionModel):
    from sklearn.svm import SVR
    model_method = SVR

    def define_model(self, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1,
                     shrinking=True, cache_size=200, verbose=False, max_iter=-1):
        """
        kernel: {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
        degree: int, default=3
        gamma: {'scale', 'auto'} or float, default='scale'
        coef0: float, default=0.0
        tol: float, default=1e-3
        C: float, default=1.0
        epsilon: float, default=0.1
        shrinking: bool, default=True
        cache_size: float, default=200
        verbose: bool, default=False
        max_iter: int, default=-1
        """
        self.model = self.model_method(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C,
                                       epsilon=epsilon, shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                                       max_iter=max_iter)

    def record_feature_contribution(self):
        if self.model.get_params()['kernel'] == 'linear':
            self.feature_contribution['coef_'] = self.model.coef_
            self.feature_contribution['intercept_'] = self.model.intercept_


# KNN
class KNN(_RegressionModel):
    from sklearn.neighbors import KNeighborsRegressor
    model_method = KNeighborsRegressor

    def define_model(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                     metric_params=None):
        """
        n_neighbors: int, default=5
        weights: {'uniform', 'distance'} or callable, default='uniform'
        algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        leaf_size: int, default=30
        p: int, default=2
        metric: str or callable, default='minkowski'
        metric_params: dict, default=None
        """
        self.model = self.model_method(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                       leaf_size=leaf_size,
                                       p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)
