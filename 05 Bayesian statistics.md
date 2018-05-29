# MLAPP 读书笔记 - 05 贝叶斯统计(Bayesian statistics)

> A Chinese Notes of MLAPP，MLAPP 中文笔记项目 
https://zhuanlan.zhihu.com/python-kivy

记笔记的人：[cycleuser](https://www.zhihu.com/people/cycleuser/activities)

2018年05月29日12:22:15



## 5.1 概论

之前咱们已经看到过很多不同类型的概率模型了,然后也讲了如何用这些模型拟合数据,比如降到了如何去进行最大后验估计(MAP)参数$\hat\theta =\arg\max p(\theta|D)$,使用各种不同先验等等还讲了全后验(full posterior)$p(\theta|D)$,以及一些特定情境下的后验预测密度(posterior predictive density)$p(x|D)$(后面的章节中会介绍更多通用算法).


对未知变量,使用后验分布来总结我们知道的信息,这就是贝叶斯统计学的核心思想.在本章要对这个统计学上的方法进行更详细讲解,在本书第六章,也就是下一章,会讲解另一种统计方法,也就是频率论统计,或者也叫做经典统计(frequentist or classical statistics).

## 5.2 总结后验分布


后验分布$p(\theta|D)$总结了我们关于未知量$\theta$的全部信息.在本节我们会讨论一些从一个概率分布,比如后验等等,能推出来的简单量.这些总结性质的统计量通常都比全面的联合分布(full joint)更好理解也更容易可视化.

### 5.2.1 最大后验估计(MAP estimation)

通过计算后验均值(mean)/中位数(median)或者众数(mode)我们可以很容易地对一个未知量进行点估计(point
 estimate).在本书5.7,会讲到如何使用决策理论(decision theory)在这些不同方法之间进行选择.通常后验均值或者中位数最适合用于对实数值变量的估计,而后验边缘分布(posterior marginals)的向量最适合对离散向量进行估计.不过后验模,也就是最大后验估计(MAP estimate)是最流行的,因为浙江问题降低到了一个优化问题(optimization problem),这样就经常能有很多高效率的算法.另外,最大后验分布还可以用非贝叶斯方式(non-Bayesian terms)来阐述,将对数先验(log prior)看作是正则化工具(regularizer)(更多内容参考本书6.5).

虽然最大后验估计的方法在计算上很有吸引力,但还是有很多缺陷的,下面就要详细讲一讲.这也为对贝叶斯方法的详尽理解提供了动力,本章后文以及本书其他地方会讲到.



此处参考原书图5.1


#### 5.2.1.1 


