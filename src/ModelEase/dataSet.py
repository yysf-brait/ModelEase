import time

import pandas as pd
from sklearn.model_selection import train_test_split

from . import decorators


class data_set:
    """
    用于存储数据集的类
    """
    data = None  # 预处理后的完整数据

    test_size = None  # 测试集比例
    random_state = None  # 随机种子

    x = None  # 自变量数据框
    y = None  # 因变量数据框

    x_train = None  # 训练集自变量
    x_test = None  # 测试集自变量
    y_train = None  # 训练集因变量
    y_test = None  # 测试集因变量

    state = None  # 数据集状态
    name = None  # 数据集名称

    def __str__(self):
        return f'{self.name} [{self.state}]'

    @decorators.cost_record('Class[Data] Init')
    def __init__(self, data: pd.DataFrame, x_index: list, y_index: int,
                 random_state: int = 0, test_size: float = 0,
                 make: bool = True,
                 name: str = 'Data ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())):
        # 初始化数据集名称
        self.name = name
        # 状态：已定义
        self.state = 'Defined'
        # 获取data的列数
        col_num = data.shape[1]
        # 判断输入的x_index是否合法
        for i in x_index:
            if i >= col_num or i < 0:
                raise ValueError(f'x_index {i} out of range')
        # 判断输入的y_index是否合法
        if y_index >= col_num or y_index < 0:
            raise ValueError(f'y_index {y_index} out of range')

        # 初始化随机种子
        self.random_state = random_state
        # 初始化测试集比例
        self.test_size = test_size

        # 初始化self.data，第一列为y，其余为x
        # 提取自变量
        x = data.iloc[:, x_index]
        # 提取因变量
        y = data.iloc[:, y_index]
        # 初始化self.data
        self.data = pd.concat([y, x], axis=1)

        # 状态：已装载
        self.state = 'Loaded'

        # 数据集准备
        if make:
            # 初始化x和y
            self.x = x
            self.y = y
            # 分割数据集
            self.x_train, self.x_test, self.y_train, self.y_test \
                = train_test_split(x,
                                   y,
                                   test_size=self.test_size,
                                   random_state=self.random_state)
            self.state = 'Made'

    # 使用data制作数据集
    @decorators.cost_record('Data_Set Make')
    def make(self):
        # 初始化self.data，第一列为y，其余为x
        # 提取自变量
        self.x = self.data.iloc[:, 1:]
        # 提取因变量
        self.y = self.data.iloc[:, 0]
        self.x_train, self.x_test, self.y_train, self.y_test \
            = train_test_split(self.x,
                               self.y,
                               test_size=self.test_size,
                               random_state=self.random_state)
        self.state = 'Made'
        return self
