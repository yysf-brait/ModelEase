# Py包示例代码说明

## 简介

这个Python包提供了一些模型和工具函数，可以帮助你进行数据预处理、模型训练和评估等任务。

## 安装

可以通过pip安装该包：

```bash
pip install -i https://test.pypi.org/simple/ ModelEase
```

## 使用示例

### 数据准备

首先，你需要准备好你的数据。在这个示例中，我们使用了一个名为`iris.csv`的数据集。

```python
import src.ModelEase as me
import pandas as pd

data1 = pd.read_csv('tests//iris.csv')
data1.fillna(data1.mean(numeric_only=True), inplace=True)
x = [0, 1, 2, 3]
y = 4
dataset = me.data_set(data1, x, y, random_state=37, test_size=0.25, make=True, name='iris')
```

### 贝叶斯模型

创建并训练贝叶斯模型：

```python
# 创建贝叶斯模型，name为模型的名字
# 请注意，这里的name是模型的名称，不是数据集的名称，并且不能与已有模型重名
# 也不能进行覆盖！TODO: 重名检测
# 默认命名为贝叶斯+日期时间，详见参数原型
Bayes = me.CNBayes(dataset, name='贝叶斯')
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
```

### 决策树模型

接着创建并训练决策树模型：

```python
# 创建决策树模型，name为模型的名字
# 请注意，这里的name是模型的名称，不是数据集的名称，并且不能与已有模型重名
Tree = me.DecisionTree(dataset, name='决策树')
# 搜索最佳参数，可设置搜索范围请见函数原型
Tree.best_params_search()
# 使用最佳参数训练模型（自动定义最佳模型）
Tree.train_best_params()
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
```
### 其他模型

其他模型的使用方法与上述两个模型基本一致，只需要将`CNBayes`和`DecisionTree`替换为其他模型即可。
目前支持的模型有：`DecisionTree`、`CNBayes`、`GNBayes`、`KNN`、`RandomForest`、`AdaBoost`、`SVM`

### 模型展示

你可以使用`me.comparison()`函数来查看注册的模型：

```python
# 你可以查看并对比已创建的模型的耗时以及效果
print(me.comparison())
# 你也可以指定me.comparison()需要展示的模型
# 使用模型的变量名作为参数
print(me.comparison(Bayes, Tree))
```

### 撤销模型

如果需要，你可以撤销已注册的模型：

```python
# 撤销模型
del Bayes

# 使用me.comparison()查看现有的模型
print(me.comparison())
# 你会发现贝叶斯模型已经被撤销
```

### 注意事项

- 指定参数进行模型训练时，请注意先定义模型再进行训练。
- 每次注册模型时，请使用不同的`name`参数，以避免混淆。
- 指定参数展示模型时，只会展示后注册的同名模型。

## 不建议的用法

### 非常规撤销模型

```python
# 当然，你也可以通过操作me.model_list来撤销模型
# 这样只会撤销模型的注册，不会删除变量或者模型本身
# 不建议进行本操作！
# 如果你再次注册同名模型，会覆盖之前模型的注册，这会导致me.comparison()无法展示之前的模型
# 并且指定参数展示模型时，若指定的模型同名，模型会覆盖之前的模型
# 例如，删除决策树模型
del me.model_list['决策树']
# 使用me.comparison()查看现有的模型
print(me.comparison())
# 你会发现决策树模型已经被撤销

# 你现在可以再注册一个同名的决策树模型了
Another_Tree = me.DecisionTree(dataset, name='决策树')
# 定义一个不一样的模型
Another_Tree.define_model(max_depth=1, min_samples_split=2, min_samples_leaf=1)
# 训练模型
Another_Tree.train()
# 使用测试集预测
Another_Tree.predict()
# 评估模型，roc参数表示是否绘制roc曲线
Another_Tree.evaluate(roc=False)
# 使用me.comparison()查看现有的模型
print(me.comparison())
# 你会发现又有了一个决策树模型

# Tree和Another_Tree是两个不同的模型
# 但是它们的名字相同，所以me.comparison()只会展示后载入的模型
print(me.comparison(Tree, Another_Tree))
# 你会发现只有Another_Tree被展示了
print(me.comparison(Another_Tree, Tree))
# 你会发现只有Tree被展示了

# 这比较复杂，所以不建议进行本操作！
# 最佳实践就是，每次注册模型时，使用不同的name参数
# 这利于你查看和对比模型
```