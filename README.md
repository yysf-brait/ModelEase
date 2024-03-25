ModelEase README

**忘记那繁琐的sklearn代码，快来试试ModelEase吧！**

# 简介

该 Python 包旨在简化机器学习模型的开发和部署过程。
它提供了高级抽象化，使得创建数据集、训练模型、进行预测以及评估模型性能的过程变得非常简单。
这极大地减少了学习成本以及编写代码的工作量。

# 适用人群

- 机器学习初学者
- 编程基础薄弱的机器学习使用者
- 需要使用机器学习的非计算机专业人士
- 需要快速开发简单机器学习模型的开发者
- 用于教学目的的机器学习教师

# 背景以及期望

在当今信息爆炸的时代，机器学习作为人工智能领域的一个重要分支，已经被广泛应用于各行各业。

然而，尽管机器学习技术的发展给我们带来了前所未有的机遇，但是对于许多人来说，学习和应用机器学习模型依然是一个艰巨的挑战。

特别是对于初级学习者来说，面对繁杂的代码和复杂的算法，往往会感到无从下手，甚至望而却步。

本 Python 包将会为这些人群提供一种简化代码、降低工作量和心智负担的解决方案。通过提供高级抽象化的模型和工具函数，使得他们可以轻松地完成创建数据集、训练模型、进行预测以及评估模型性能的全过程。

我们期望这个包能够帮助更多的人进入机器学习领域，享受到机器学习技术带来的乐趣和成就感。

# 亮点

1. 统一且简洁的API
2. 便于快速部署且易于使用的封装
3. 一键评估模型效果、一键对比模型性能

# 主要功能

## 数据集

1. 根据给定的数据集、特征列和目标列，创建数据集。

## 模型

使用面对对象的方式，只需要几行代码就可以完成如下的工作：

1. 创建模型
2. 定义模型
3. 训练模型
4. (搜索最佳参数)
5. (使用最佳参数训练模型)
6. 预测测试集
7. 评估模型
    - 准确率
    - 精确率
    - 召回率
    - F1 值
    - 混淆矩阵
    - ROC 曲线
8. 对比模型的效果
    - 训练耗时
    - 预测耗时
    - 准确率
    - 精确率
    - 召回率
    - F1 值

# API

- 创建数据集 `me.dataSet.data_set`
-

创建模型 `me.model.CNBayes`, `me.model.DecisionTree`, `me.model.GNBayes`, `me.model.KNN`, `me.model.RandomForest`, `me.model.AdaBoost`, `me.model.SVM`

- 定义模型 `a_model.define_model`
- 训练模型 `a_model.train`
- 搜索最佳参数 `a_model.best_params_search`
- 使用最佳参数训练模型 `a_model.train_best_params`
- 预测测试集 `a_model.predict`
- 评估模型 `a_model.evaluate`
- 对比模型的效果 `me.model.comparison`

| me           | sklearn                         |
|--------------|---------------------------------|
| CNBayes      | naive_bayes.CategoricalNB       |
| DecisionTree | tree.DecisionTreeClassifier     |
| GNBayes      | naive_bayes.GaussianNB          |
| KNN          | neighbors.KNeighborsClassifier  |
| RandomForest | ensemble.RandomForestClassifier |
| AdaBoost     | ensemble.AdaBoostClassifier     |
| SVM          | svm.SVC                         |

# TODO

- 内置的数据集处理方法
- 更多的模型
- 更人性化、简洁的 API

# 安装

本包尚未发布到 PyPI，但是您可以通过以下方式安装：

```bash
pip install -i https://test.pypi.org/simple/ ModelEase
```

# 使用示例

以下是一些简单的示例，展示了如何使用本包中的模型和工具函数。

## 数据准备

在开始使用之前，请确保您已准备好相应的数据，并确保已经做好了数据预处理工作。

```python
import src.ModelEase as me
import pandas as pd

data = pd.read_csv('tests//iris.csv')
data.fillna(data.mean(numeric_only=True), inplace=True)
```

## 创建data_set

创建data_set前，需要先定义特征列和目标列

```python
# 设置特征列和目标列
features = [0, 1, 2, 3]
target = 4
```

接下来我们创建一个名为 iris 的数据集，其中测试集的比例为 0.25。
设置 random_state 为 37，以确保每次运行时都能得到相同的结果。
make 参数指定为 True，指示不需要额外数据预处理，直接执行数据集创建操作。

```python
# 创建数据集
dataset = me.dataSet.data_set(data, features, target, random_state=37, test_size=0.25, make=True, name='iris')
```

## 创建模型

### 贝叶斯模型

接下来，让我们创建并训练一个贝叶斯模型。
我们使用 dataset 来创建模型，设置模型的 name 为“贝叶斯”。
创建模型时，模型会自动注册到 me.model.model_list 中。

```python
# 创建贝叶斯模型
Bayes = me.model.CNBayes(dataset, name='贝叶斯')
```

随后使用最佳参数搜索功能来搜索最佳参数，并使用这些参数来训练模型。

```python
# 搜索最佳参数
Bayes.best_params_search()
# 使用最佳参数训练模型
Bayes.train_best_params()
```

最后，我们将会使用测试集来评估模型的性能。

```python
# 预测测试集
Bayes.predict()
# 评估模型
Bayes.evaluate(roc=False)
# 绘制 ROC 曲线
Bayes.roc()
```

我们会得到模型的准确率、精确率、召回率、F1 值，以及混淆矩阵和 ROC 曲线。

### 决策树模型

接下来，让我们创建并训练一个决策树模型。
我们使用 dataset 来创建模型，设置模型的 name 为“决策树”。
创建模型时，模型会自动注册到 me.model.model_list 中。

```python
# 创建决策树模型
Tree = me.model.DecisionTree(dataset, name='决策树')
```

这次我们将使用指定参数来训练模型。
我们需要先定义模型，再训练模型。

```python
# 定义模型
Tree.define_model(criterion='gini', max_depth=4, min_samples_split=2, min_samples_leaf=2)
# 训练模型
Tree.train()
```

最后，我们将会使用测试集来评估模型的性能。

```python
# 预测测试集
Tree.predict()
# 评估模型
Tree.evaluate(roc=False)
# 绘制 ROC 曲线
Tree.roc()
```

我们会得到模型的准确率、精确率、召回率、F1 值，以及混淆矩阵和 ROC 曲线。

### 其他模型

除了上述两个模型外，我们还支持以下其他模型的使用方法：GNBayes、KNN、RandomForest、AdaBoost、SVM。
调用这些模型的方法与上述两个模型完全相同，你只需要调用相应的类即可。

## 模型展示

您可以使用 me.model.comparison() 函数查看并对比模型。
你会得到一个表格，其中包含了已注册的模型及其训练耗时、预测耗时、准确率、精确率、召回率、F1 值。

```python
# 查看已注册的模型及其性能
print(me.model.comparison())
# 您也可以指定需要展示的模型
print(me.model.comparison(Bayes, Tree))
```

## 撤销模型

如果需要，您可以撤销已注册的模型：

```python
# 撤销模型
del Bayes
```

```python
# 使用 me.model.comparison() 查看现有的模型
print(me.model.comparison())
# 您会发现贝叶斯模型已经被撤销
```

# 注意事项

- 在指定参数进行模型训练时，请先定义模型再进行训练。
- 每次注册模型时，请使用不同的 name 参数，以避免混淆。
- 在指定参数展示模型时，只会展示后注册的同名模型。

# 不建议的用法

## 非常规撤销模型

通过操作 me.model.model_list 来撤销模型是不建议的
这样只会撤销模型的注册，不会删除变量或者模型本身
如果您再次注册同名模型，会覆盖之前模型的注册，导致展示和对比时的混淆

```python
# 例如，删除决策树模型
del me.model.model_list['决策树']
# 使用 me.model.comparison() 查看现有的模型
print(me.model.comparison())
# 您会发现决策树模型已经被撤销

# 您现在可以再注册一个同名的决策树模型了
Another_Tree = me.model.DecisionTree(dataset, name='决策树')
# 定义一个不一样的模型
Another_Tree.define_model(max_depth=1, min_samples_split=2, min_samples_leaf=1)
# 训练模型
Another_Tree.train()
# 预测测试集
Another_Tree.predict()
# 评估模型
Another_Tree.evaluate(roc=False)
# 使用 me.model.comparison() 查看现有的模型
print(me.model.comparison())
# 您会发现又有了一个决策树模型

# Tree 和 Another_Tree 是两个不同的模型
# 但是它们的名字相同，所以 me.model.comparison() 只会展示后载入的模型
print(me.model.comparison(Tree, Another_Tree))
# 您会发现只有 Another_Tree 被展示了
print(me.model.comparison(Another_Tree, Tree))
# 您会发现只有 Tree 被展示了

# 这比较复杂，所以不建议进行此操作！
```

最佳实践是每次注册模型时，使用不同的 name 参数，以便于查看和对比模型
