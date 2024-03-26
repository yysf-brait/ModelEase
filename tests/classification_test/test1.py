import pandas as pd

import src.ModelEase as me # noqa

# 请注意路径是否正确
data1 = pd.read_csv('tests//iris.csv')
# 尚未创建任何数据预处理方法，请注意在创建data_set前自行进行数据预处理
data1.fillna(data1.mean(numeric_only=True), inplace=True)
x = [0, 1, 2, 3]
y = 4
# data: 数据集, x_index: 自变量索引, y_index: 因变量索引
# random_state: 随机种子, test_size: 测试集比例
# make: 是否创建数据集（分割x、y的测试集与训练集）, name: 数据集名称
dataset = me.dataSet.data_set(data1, x, y, random_state=37, test_size=0.25, make=True, name='iris')

# 创建贝叶斯模型，name为模型的名字
# 请注意，这里的name是模型的名称，不是数据集的名称，并且不能与已有模型重名
# 默认命名为贝叶斯+日期时间，详见参数原型
Bayes = me.model.CNBayes(dataset, name='贝叶斯')
# 搜索最佳参数，可设置搜索范围请见函数原型
Bayes.best_params_search()
# 使用最佳参数训练模型（自动定义最佳模型）
Bayes.train_best_params()
# 当然你也可以直接使用自定义参数训练模型
# 但是请注意，需要先定义模型，再训练模型
# Bayes.define_model(alpha=1.0, fit_prior=True)
# Bayes.train()
# 使用测试集预测
Bayes.predict()
# 评估模型，roc参数表示是否绘制roc曲线
Bayes.evaluate(roc=False)
# 也可以直接使用roc参数绘制roc曲线
Bayes.roc()

# 创建决策树模型，name为模型的名字
# 请注意，这里的name是模型的名称，不是数据集的名称，并且不能与已有模型重名
Tree = me.model.DecisionTree(dataset, name='决策树')
# 搜索最佳参数，可设置搜索范围请见函数原型
Tree.define_model(criterion='gini', max_depth=4, min_samples_split=2, min_samples_leaf=2)
# 使用最佳参数训练模型（自动定义最佳模型）
Tree.train()
# 当然你也可以直接使用自定义参数训练模型
# 但是请注意，需要先定义模型，再训练模型
# Tree.define_model(max_depth=3, min_samples_split=2, min_samples_leaf=1)
# Tree.train()
# 使用测试集预测
Tree.predict()
# 评估模型，roc参数表示是否绘制roc曲线
Tree.evaluate(roc=False)
# 也可以直接使用roc参数绘制roc曲线
Tree.roc()

# 其他模型的使用方法与上述两个模型基本一致

# 你可以查看并对比已创建的模型的耗时以及效果
print(me.comparison())


# 撤销模型
del Bayes


# 当然，你也可以通过操作me.model_list来撤销模型
# 这样只会撤销模型的注册，不会删除变量或者模型本身
# 不建议进行本操作！
# 如果你再次注册同名模型，会覆盖之前模型的注册，这会导致me.show()无法展示之前的模型
# 并且指定参数展示模型时，若指定的模型同名，模型会覆盖之前的模型
# 例如，删除决策树模型
del me.table['决策树']
# 使用me.show()查看现有的模型
print(me.comparison())
# 你会发现决策树模型已经被撤销

# 你现在可以再注册一个同名的决策树模型了
Another_Tree = me.model.DecisionTree(dataset, name='决策树')
# 定义一个不一样的模型
Another_Tree.define_model(max_depth=1, min_samples_split=2, min_samples_leaf=1)
# 训练模型
Another_Tree.train()
# 使用测试集预测
Another_Tree.predict()
# 评估模型，roc参数表示是否绘制roc曲线
Another_Tree.evaluate(roc=False)
# 使用me.show()查看现有的模型
print(me.comparison())
# 你会发现又有了一个决策树模型



# 这比较复杂，所以不建议进行本操作！
# 最佳实践就是，每次注册模型时，使用不同的name参数
# 这利于你查看和对比模型
