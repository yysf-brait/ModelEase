import pandas as pd
from sklearn.datasets import load_iris

# 加载iris数据集
iris = load_iris()

# 将数据集转换为DataFrame格式
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target  # 添加target列

# 保存数据集
iris_df.to_csv('tests//classification_test//iris.csv', index=False)
