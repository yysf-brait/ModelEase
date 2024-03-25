import src.ModelEase as me # noqa
import pandas as pd

# 读入calibration数据集
df = pd.read_csv('tests//regression_test//california.csv')

# 创建数据集
data = me.data_set(df, x_index=[0, 1, 2, 3, 4, 5, 6, 7], y_index=8, test_size=0.2, name='California Housing')

# LinearRegression
model = me.model.regression.LinearRegression(data, name='LinearRegression')
model.define_model()
model.train()
model.predict()
model.scatter()
model.evaluate()

model.auto()

