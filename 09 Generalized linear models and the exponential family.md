# MLAPP 读书笔记 - 09 通用线性模型(Generalized linear models)和指数族分布(exponential family)

> A Chinese Notes of MLAPP,MLAPP 中文笔记项目 
https://zhuanlan.zhihu.com/python-kivy

记笔记的人：[cycleuser](https://www.zhihu.com/people/cycleuser/activities)

2018年6月27日09:26:15

## 9.1 概论

之前已经见到过很多概率分布了:正态(高斯)分布,伯努利分布(Bernoulli),学生T分布,均匀分布,$\gamma$分布等等.这些大多数都属于指数族分布(exponential family).本章就要讲这类分布的各种特点.然后我们就能用来推出很多广泛应用的定力和算法.

接下来我们会看到要构建一个生成分类器如何简单地用指数族分布中的某个成员来作为类条件密度.另外还会讲到如何构建判别模型,其中响应变量服从指数族分布,均值是输入特征的线性函数;这就叫做通用线性模型(Generalized linear models),将逻辑回归上的思想扩展到了其他类别的响应变量上去.