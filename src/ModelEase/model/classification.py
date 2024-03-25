import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import decorators
from .. import model_list
from .. import n_jobs
from ..dataSet import data_set


def comparison(*args) -> pd.DataFrame:
    # 判断是否有参数
    if len(args) == 0:
        return pd.DataFrame(model_list).T
    else:
        ret = {}
        for i in args:
            ret[i.name] = {'train_cost': i.Train_Cost if hasattr(i, 'train_cost') else None,
                           'predict_cost': i.Predict_Cost if hasattr(i, 'predict_cost') else None,
                           'accuracy': i.accuracy if hasattr(i, 'accuracy') else None,
                           'precision': i.precision if hasattr(i, 'precision') else None,
                           'recall': i.recall if hasattr(i, 'recallpip') else None,
                           'f1_score': i.f1_score if hasattr(i, 'f1_score') else None}
    return pd.DataFrame(ret).T


class _ClassificationModel:
    """
    模型的基类
    """
    model = None  # 模型
    name = None  # 模型名称
    model_method = None  # 模型方法
    best_params = None  # 最佳参数

    random_state = None  # 随机种子
    data_name = None  # 数据集名称
    x_train = None  # 训练集自变量
    x_test = None  # 测试集自变量
    y_train = None  # 训练集因变量
    y_test = None  # 测试集因变量

    train_cost = None  # 训练耗时
    predict_cost = None  # 预测耗时

    y_pred = None  # 预测结果
    accuracy = None  # 准确率
    precision = None  # 精确率
    recall = None  # 召回率
    f1_score = None  # F1分数

    def __str__(self):
        return f'{self.name} [{self.data_name}]'

    def __init__(self, data: data_set, name: str = 'Model', random_state: int = None):
        self.name = name
        # 如果模型名称已存在
        if self.name in model_list:
            raise ValueError('Model name exists. You can use default name or another name.')
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

    # 搜索最佳参数
    def best_params_search(self, **param_grid):
        from sklearn.model_selection import GridSearchCV
        grid = GridSearchCV(self.model_method(), param_grid, refit=False, verbose=1, n_jobs=n_jobs)
        grid.fit(self.x_train, self.y_train)
        self.best_params = grid.best_params_
        print('best params:', self.best_params)

    # 定义模型
    def define_model(self, **kwargs):
        self.model = self.model_method(**kwargs)

    # 使用最佳参数定义模型
    def define_best_params_model(self):
        self.define_model(**self.best_params)

    # 训练模型
    def train(self):
        self.model.fit(self.x_train, self.y_train)

    # 训练最佳参数
    def train_best_params(self):
        self.define_best_params_model()
        self.train()

    # 预测
    def predict(self):
        self.y_pred = self.model.predict(self.x_test)

    # 评估模型
    def evaluate(self, roc: bool = False):
        # 评估模型
        from sklearn import metrics
        import seaborn as sns
        # 生成混淆矩阵
        cm = pd.crosstab(self.y_pred, self.y_test)
        sns.heatmap(cm, annot=True, cmap=r'GnBu', fmt=r'd')
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')
        plt.show()
        # 输出分类报告
        print(metrics.classification_report(self.y_test, self.y_pred))

        # 计算准确率、精确率、召回率、F1分数
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        self.precision = metrics.precision_score(self.y_test, self.y_pred, average='weighted')
        self.recall = metrics.recall_score(self.y_test, self.y_pred, average='weighted')
        self.f1_score = metrics.f1_score(self.y_test, self.y_pred, average='weighted')
        # 更新全局变量model_list
        model_list[self.name] = dict(
            accuracy=self.accuracy,
            precision=self.precision,
            recall=self.recall,
            f1_score=self.f1_score
        )
        # 输出评估结果
        print(f'accuracy: {self.accuracy}')
        print(f'precision: {self.precision}')
        print(f'recall: {self.recall}')
        print(f'f1_score: {self.f1_score}')
        if roc:
            self.roc()

    # ROC曲线
    def roc(self):
        # 检查是否是二分类问题
        if (n_classes := len(self.y_test.unique())) == 2:
            if hasattr(self.model, "predict_proba"):
                from sklearn.metrics import roc_curve, auc
                # 获取预测概率
                y_score = self.model.predict_proba(self.x_test)[:, 1]
                # 计算fpr, tpr, thresholds
                fpr, tpr, thresholds = roc_curve(self.y_test, y_score)
                # 计算AUC
                roc_auc = auc(fpr, tpr)
                # 绘制ROC曲线
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.show()
            else:
                print('no predict_proba, no ROC')
        else:
            from sklearn.metrics import roc_curve, auc
            # 获取测试集上的预测概率
            y_score = self.model.predict_proba(self.x_test)
            # 计算ROC曲线
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(self.y_test, y_score[:, i], pos_label=self.model.classes_[i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # 绘制ROC曲线
            plt.figure()
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'black', 'pink', 'brown']
            for i, color in zip(range(n_classes), colors[0:n_classes]):
                # 随机选择颜色
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(self.model.classes_[i], roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.show()


# 决策树
class DecisionTree(_ClassificationModel):
    from sklearn import tree
    model_method = tree.DecisionTreeClassifier

    # 定义构造函数
    @decorators.cost_record('Class[DecisionTree] Init')
    def __init__(self, data: data_set,
                 name: str = None,
                 random_state: int = None):
        if name is None:
            name = 'DecisionTree' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        super().__init__(data, name, random_state)

    # 搜索最佳参数
    @decorators.cost_record('DecisionTree-BestParamsSearch')
    def best_params_search(self, criterion: List[str] = None,
                           max_depth: List[int] = None,
                           min_samples_split: List[int] = None,
                           min_samples_leaf: List[int] = None):
        if criterion is None:
            criterion = ['gini', 'entropy']
        if max_depth is None:
            max_depth = [2, 3, 4, 5, 6]
        if min_samples_split is None:
            min_samples_split = [2, 4, 6, 8]
        if min_samples_leaf is None:
            min_samples_leaf = [2, 4, 8, 10, 12]
        super().best_params_search(criterion=criterion,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf)

    # 定义模型
    @decorators.cost_record('DecisionTree-DefineModel')
    def define_model(self, criterion: str = 'gini', max_depth: int = None, min_samples_split: int = 2,
                     min_samples_leaf: int = 1, random_state: int = None):
        super().define_model(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf, random_state=random_state)

    # 使用最佳参数定义模型
    @decorators.cost_record('DecisionTree-DefineBestParamsModel')
    def define_best_params_model(self):
        super().define_best_params_model()

    # 训练模型
    @decorators.cost_record('DecisionTree-Train')
    def train(self):
        super().train()

    # 训练最佳参数
    def train_best_params(self):
        super().train_best_params()

    # 预测
    @decorators.cost_record('DecisionTree-Predict')
    def predict(self):
        super().predict()

    # 评估模型
    @decorators.cost_record('DecisionTree-Evaluate')
    def evaluate(self, roc: bool = False):
        super().evaluate(roc)

    # ROC曲线
    @decorators.cost_record('DecisionTree-ROC')
    def roc(self):
        super().roc()


# 朴素贝叶斯
class CNBayes(_ClassificationModel):
    from sklearn import naive_bayes
    model_method = naive_bayes.CategoricalNB

    # 定义构造函数
    @decorators.cost_record('Class[CNBayes] Init')
    def __init__(self, data: data_set,
                 name: str = None,
                 random_state: int = None):
        if name is None:
            name = 'CNBayes' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        super().__init__(data, name, random_state)

    # 搜索最佳参数
    @decorators.cost_record('CNBayes-BestParamsSearch')
    def best_params_search(self, alpha: List[float] = None, fit_prior: List[bool] = None):
        if alpha is None:
            alpha = [0.1, 0.5, 0.1, 1.0, 2.0]
        if fit_prior is None:
            fit_prior = [True, False]
        super().best_params_search(alpha=alpha, fit_prior=fit_prior)

    # 定义模型
    @decorators.cost_record('CNBayes-DefineModel')
    def define_model(self, alpha: float = 1.0, fit_prior: bool = True):
        super().define_model(alpha=alpha, fit_prior=fit_prior)

    # 使用最佳参数定义模型
    @decorators.cost_record('CNBayes-DefineBestParamsModel')
    def define_best_params_model(self):
        super().define_best_params_model()

    # 训练模型
    @decorators.cost_record('CNBayes-Train')
    def train(self):
        super().train()

    # 训练最佳参数
    def train_best_params(self):
        super().train_best_params()

    # 预测
    @decorators.cost_record('CNBayes-Predict')
    def predict(self):
        super().predict()

    # 评估模型
    @decorators.cost_record('CNBayes-Evaluate')
    def evaluate(self, roc: bool = False):
        super().evaluate(roc)

    # ROC曲线
    @decorators.cost_record('CNBayes-ROC')
    def roc(self):
        super().roc()


# 高斯朴素贝叶斯
class GNBayes(_ClassificationModel):
    from sklearn import naive_bayes
    model_method = naive_bayes.GaussianNB

    # 定义构造函数
    @decorators.cost_record('Class[GNBayes] Init')
    def __init__(self, data: data_set,
                 name: str = None,
                 random_state: int = None):
        if name is None:
            name = 'GNBayes' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        super().__init__(data, name, random_state)

    # 搜索最佳参数
    @decorators.cost_record('GNBayes-BestParamsSearch')
    def best_params_search(self, var_smoothing: List[float] = None):
        if var_smoothing is None:
            var_smoothing = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        super().best_params_search(var_smoothing=var_smoothing)

    # 定义模型
    @decorators.cost_record('GNBayes-DefineModel')
    def define_model(self):
        super().define_model()

    # 使用最佳参数定义模型
    @decorators.cost_record('GNBayes-DefineBestParamsModel')
    def define_best_params_model(self):
        super().define_best_params_model()

    # 训练模型
    @decorators.cost_record('GNBayes-Train')
    def train(self):
        super().train()

    # 训练最佳参数
    def train_best_params(self):
        super().train_best_params()

    # 预测
    @decorators.cost_record('GNBayes-Predict')
    def predict(self):
        super().predict()

    # 评估模型
    @decorators.cost_record('GNBayes-Evaluate')
    def evaluate(self, roc: bool = False):
        super().evaluate(roc)

    # ROC曲线
    @decorators.cost_record('GNBayes-ROC')
    def roc(self):
        super().roc()


# KNN
class KNN(_ClassificationModel):
    from sklearn import neighbors
    model_method = neighbors.KNeighborsClassifier

    # 定义构造函数
    @decorators.cost_record('Class[KNN] Init')
    def __init__(self, data: data_set,
                 name: str = None,
                 random_state: int = None):
        if name is None:
            name = 'KNN' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        super().__init__(data, name, random_state)

    # 搜索最佳参数
    @decorators.cost_record('KNN-BestParamsSearch')
    def best_params_search(self, n_neighbors: List[int] = None, weights: List[str] = None,
                           algorithm: List[str] = None, leaf_size: List[int] = None, p: List[int] = None):
        if n_neighbors is None:
            n_neighbors = [i for i in
                           range(1, min(20, int(np.ceil(np.log2(self.x_test.shape[0] + self.x_train.shape[0])))))]
        if weights is None:
            weights = ['uniform', 'distance']
        if algorithm is None:
            algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        if leaf_size is None:
            leaf_size = [10, 20, 30, 40, 50]
        if p is None:
            p = [1, 2]
        super().best_params_search(n_neighbors=n_neighbors, weights=weights,
                                   algorithm=algorithm, leaf_size=leaf_size, p=p)

    # 定义模型
    @decorators.cost_record('KNN-DefineModel')
    def define_model(self, n_neighbors: int = 5, weights: str = 'uniform', algorithm: str = 'auto',
                     leaf_size: int = 30, p: int = 2):
        super().define_model(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                             leaf_size=leaf_size, p=p)

    # 使用最佳参数定义模型
    @decorators.cost_record('KNN-DefineBestParamsModel')
    def define_best_params_model(self):
        super().define_best_params_model()

    # 训练模型
    @decorators.cost_record('KNN-Train')
    def train(self):
        super().train()

    # 训练最佳参数
    def train_best_params(self):
        super().train_best_params()

    # 预测
    @decorators.cost_record('KNN-Predict')
    def predict(self):
        super().predict()

    # 评估模型
    @decorators.cost_record('KNN-Evaluate')
    def evaluate(self, roc: bool = False):
        super().evaluate(roc)

    # ROC曲线
    @decorators.cost_record('KNN-ROC')
    def roc(self):
        super().roc()


# 随机森林
class RandomForest(_ClassificationModel):
    from sklearn import ensemble
    model_method = ensemble.RandomForestClassifier

    # 定义构造函数
    @decorators.cost_record('Class[RandomForest] Init')
    def __init__(self, data: data_set,
                 name: str = None,
                 random_state: int = None):
        if name is None:
            name = 'RandomForest' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        super().__init__(data, name, random_state)

    # 搜索最佳参数
    @decorators.cost_record('RandomForest-BestParamsSearch')
    def best_params_search(self, n_estimators: List[int] = None, criterion: List[str] = None,
                           max_depth: List[int] = None, min_samples_split: List[int] = None,
                           min_samples_leaf: List[int] = None, max_features: List[str] = None,
                           random_state: List[int] = None):
        if n_estimators is None:
            n_estimators = [100, 200, 300, 400]
        if criterion is None:
            criterion = ['gini', 'entropy']
        if max_depth is None:
            max_depth = [2, 3, 5, None]
        if min_samples_split is None:
            min_samples_split = [2, 4, 6]
        if min_samples_leaf is None:
            min_samples_leaf = [1, 2, 4]
        if max_features is None:
            max_features = ['sqrt', 'log2', None]
        if random_state is None:
            random_state = [self.random_state]
        super().best_params_search(n_estimators=n_estimators, criterion=criterion,
                                   max_depth=max_depth, min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf, max_features=max_features,
                                   random_state=random_state)

    # 定义模型
    @decorators.cost_record('RandomForest-DefineModel')
    def define_model(self, n_estimators: int = 100, criterion: str = 'gini', max_depth: int = None,
                     min_samples_split: int = 2, min_samples_leaf: int = 1, max_features: str = 'auto',
                     random_state: int = None):
        super().define_model(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                             max_features=max_features, random_state=random_state)

    # 使用最佳参数定义模型
    @decorators.cost_record('RandomForest-DefineBestParamsModel')
    def define_best_params_model(self):
        super().define_best_params_model()

    # 训练模型
    @decorators.cost_record('RandomForest-Train')
    def train(self):
        super().train()

    # 训练最佳参数
    def train_best_params(self):
        super().train_best_params()

    # 预测
    @decorators.cost_record('RandomForest-Predict')
    def predict(self):
        super().predict()

    # 评估模型
    @decorators.cost_record('RandomForest-Evaluate')
    def evaluate(self, roc: bool = False):
        super().evaluate(roc)

    # ROC曲线
    @decorators.cost_record('RandomForest-ROC')
    def roc(self):
        super().roc()


# Adaboost
class AdaBoost(_ClassificationModel):
    from sklearn import ensemble
    from sklearn.tree import DecisionTreeClassifier
    model_method = ensemble.AdaBoostClassifier

    # 定义构造函数
    @decorators.cost_record('Class[AdaBoost] Init')
    def __init__(self, data: data_set,
                 name: str = None,
                 random_state: int = None):
        if name is None:
            name = 'AdaBoost' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        super().__init__(data, name, random_state)

    # 搜索最佳参数
    @decorators.cost_record('AdaBoost-BestParamsSearch')
    def best_params_search(self, estimator: List[DecisionTreeClassifier] = None,
                           n_estimators: List[int] = None,
                           learning_rate: List[float] = None,
                           random_state: List[int] = None):
        from sklearn.tree import DecisionTreeClassifier
        if estimator is None:
            estimator = [DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=3)]
        if n_estimators is None:
            n_estimators = [50, 100, 200, 300]
        if learning_rate is None:
            learning_rate = [0.1, 0.5, 0.8, 1]
        if random_state is None:
            random_state = [self.random_state]
        super().best_params_search(estimator=estimator, n_estimators=n_estimators,
                                   learning_rate=learning_rate, random_state=random_state, algorithm=['SAMME'])

    # 定义模型
    @decorators.cost_record('AdaBoost-DefineModel')
    def define_model(self, estimator: DecisionTreeClassifier = None,
                     n_estimators: int = 50,
                     learning_rate: float = 1.0,
                     random_state: int = None):
        super().define_model(estimator=estimator,
                             n_estimators=n_estimators,
                             learning_rate=learning_rate,
                             random_state=random_state,
                             algorithm='SAMME')

    # 使用最佳参数定义模型
    @decorators.cost_record('AdaBoost-DefineBestParamsModel')
    def define_best_params_model(self):
        super().define_best_params_model()

    # 训练模型
    @decorators.cost_record('AdaBoost-Train')
    def train(self):
        super().train()

    # 训练最佳参数
    def train_best_params(self):
        super().train_best_params()

    # 预测
    @decorators.cost_record('AdaBoost-Predict')
    def predict(self):
        super().predict()

    # 评估模型
    @decorators.cost_record('AdaBoost-Evaluate')
    def evaluate(self, roc: bool = False):
        super().evaluate(roc)

    # ROC曲线
    @decorators.cost_record('AdaBoost-ROC')
    def roc(self):
        super().roc()


# 向量机
class SVM(_ClassificationModel):
    from sklearn import svm
    model_method = svm.SVC

    # 定义构造函数
    @decorators.cost_record('Class[SVM] Init')
    def __init__(self, data: data_set,
                 name: str = None,
                 random_state: int = None):
        if name is None:
            name = 'SVM' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        super().__init__(data, name, random_state)

    # 搜索最佳参数
    @decorators.cost_record('SVM-BestParamsSearch')
    def best_params_search(self, c: List[float] = None, kernel: List[str] = None,
                           degree: List[int] = None, gamma: List[str] = None,
                           random_state: List[int] = None):
        if c is None:
            c = [0.1, 1, 10]
        if kernel is None:
            kernel = ['rbf']
        if degree is None:
            degree = [3, 4, 5]
        if gamma is None:
            gamma = ['scale', 'auto']
        if random_state is None:
            random_state = [self.random_state]
        super().best_params_search(C=c, kernel=kernel, degree=degree,
                                   gamma=gamma, random_state=random_state,
                                   probability=[True])

    # 定义模型
    @decorators.cost_record('SVM-DefineModel')
    def define_model(self, c: float = 1.0, kernel: str = 'rbf', degree: int = 3, gamma: str = 'scale',
                     random_state: int = None):
        super().define_model(C=c, kernel=kernel, degree=degree, gamma=gamma, random_state=random_state,
                             probability=True)

    # 使用最佳参数定义模型
    @decorators.cost_record('SVM-DefineBestParamsModel')
    def define_best_params_model(self):
        super().define_best_params_model()

    # 训练模型
    @decorators.cost_record('SVM-Train')
    def train(self):
        super().train()

    # 训练最佳参数
    def train_best_params(self):
        super().train_best_params()

    # 预测
    @decorators.cost_record('SVM-Predict')
    def predict(self):
        super().predict()

    # 评估模型
    @decorators.cost_record('SVM-Evaluate')
    def evaluate(self, roc: bool = False):
        super().evaluate(roc)

    # ROC曲线
    @decorators.cost_record('SVM-ROC')
    def roc(self):
        super().roc()
