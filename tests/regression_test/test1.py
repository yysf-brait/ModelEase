import pandas as pd

import src.ModelEase as me  # noqa

# 读入calibration数据集
df = pd.read_csv('tests//regression_test//california.csv')

# 创建数据集
data = me.data_set(df, x_index=[0, 1, 2, 3, 4, 5, 6, 7], y_index=8, test_size=0.2, name='California Housing')

Linear = me.model.regression.LinearRegression(data, name='Linear Regression')
Linear.define_model()
Linear.train()
Linear.predict()
Linear.scatter()
print(Linear.coef, Linear.intercept)
Linear.evaluate()

Linear.auto()

Ridge = me.model.regression.RidgeRegression(data, name='Ridge Regression')
Ridge.define_model(alpha=0.5)
Ridge.train()
Ridge.predict()
Ridge.scatter()
print(Ridge.coef, Ridge.intercept)
Ridge.evaluate()

Ridge.auto()
