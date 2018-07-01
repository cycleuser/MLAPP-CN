# MLAPP 读书笔记 - 10 离散图模型(Directed graphical models)(贝叶斯网络(Bayes nets))

> A Chinese Notes of MLAPP,MLAPP 中文笔记项目 
https://zhuanlan.zhihu.com/python-kivy

记笔记的人：[cycleuser](https://www.zhihu.com/people/cycleuser/activities)

2018年7月1日18:59:12

## 10.1 概论

>以简单方式对待复杂系统的原则,我基本知道两个:首先就是模块化原则,其次就是抽象原则.我是机器学习中计算概率的辩护者,因为我相信概率论对这两种原则都有深刻又有趣的实现方式,分别通过可分解性和平均.在我看来,尽可能充分利用这两种机制,是机器学习的前进方向.                      Michael Jordan, 1997 (转引自 (Frey 1998)).

假如我们观测多组相关变量,比如文档中的词汇,或者图像中的像素,再或基因片段上的基因.怎么能简洁地表示联合分布$p(x|\theta)$呢?利用这个分布,给定其他变量情况下,怎么能以合理规模的计算时间来推导一系列的变量呢?怎么通过适当规模的数据来学习得到这个分布的参数呢?这些问题就是概率建模(probabilistic modeling),推导(inference)和学习(learning)的核心,也是本章主题了.

### 10.1.1 链式规则(Chain rule)