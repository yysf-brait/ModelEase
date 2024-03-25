import pandas as pd
from sklearn.datasets import fetch_california_housing

# 加载California housing数据集
california_housing = fetch_california_housing()

# 将数据集转换为DataFrame格式
california_df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_df['PRICE'] = california_housing.target  # 添加房价列

# 保存数据集
california_df.to_csv('tests//regression_test//california.csv', index=False)
