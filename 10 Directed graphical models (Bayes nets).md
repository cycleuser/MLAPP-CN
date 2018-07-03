# MLAPP 读书笔记 - 10 离散图模型(Directed graphical models)(贝叶斯网络(Bayes nets))

> A Chinese Notes of MLAPP,MLAPP 中文笔记项目 
https://zhuanlan.zhihu.com/python-kivy

记笔记的人：[cycleuser](https://www.zhihu.com/people/cycleuser/activities)

2018年7月1日18:59:12

## 10.1 概论

>以简单方式对待复杂系统的原则,我基本知道两个:首先就是模块化原则,其次就是抽象原则.我是机器学习中计算概率的辩护者,因为我相信概率论对这两种原则都有深刻又有趣的实现方式,分别通过可分解性和平均.在我看来,尽可能充分利用这两种机制,是机器学习的前进方向.                      Michael Jordan, 1997 (转引自 (Frey 1998)).

假如我们观测多组相关变量,比如文档中的词汇,或者图像中的像素,再或基因片段上的基因.怎么能简洁地表示联合分布$p(x|\theta)$呢?利用这个分布,给定其他变量情况下,怎么能以合理规模的计算时间来推导一系列的变量呢?怎么通过适当规模的数据来学习得到这个分布的参数呢?这些问题就是概率建模(probabilistic modeling),推导(inference)和学习(learning)的核心,也是本章主题了.

### 10.1.1 链式规则(Chain rule)

通过概率论的链式规则,就可以讲一个联合分布写成下面的形式,使用任意次序的变量都可以:

$p(x_{1:V} ) = p(x_1)p(x_2|x_1)p(x_3|x_2, x_1)p(x_4|x_1, x_2, x_3) ... p(x_V |x_{1:V −1})$(10.1)

其中的V是变量数目,MATLAB风格的记号$1:V$表示集合{1,2,...,V},为了简洁,上面狮子中去掉了对固定参数$\theta$的条件.这个表达式的问题在于随着T变大,要表示条件分布$p(x_t|x_{1:t-1})$就越来越复杂了.

例如,设所有变量都有K个状态(states).然后可以将$p(x_1)$表示为$O(K)$个数字的表格,表示了一个离散分布(实际上只有K-1个自由参数,因为总和为一的约束条件,不过这里为了写起来简单就写成O(K)了).然后还可以将$p(x_2|x_1)$写成一个$O(K^2)$个数值的表格,写出$p(x_2=j|x_1=i)=T_{ij}$;T叫做随机矩阵(stochastic matrix)对$0\le T+{ij}\le 1$的所有条目以及所有列i满足约束条件$\sum_jT_{ij}=1$.与此类似,还可以将$p(x_3|x_1, x_2)$表示成一个有$O(K^3)$个数值的三维表格.这些表格就叫做条件概率表格(conditional probability tables,缩写为CPT).然后在我们这个模型里面就有$O(K^V)$个参数.要学习得到这么多参数需要的数据量就要多得可怕了.

要解决这个问题可以将每个条件概率表(CPT)替换成条件概率分布(conditional probability distribution,缩写为CPD),比如多想逻辑回归,也就是$p(x_t=k|x_{1:t-1})=S(W_tx_{1:t-1})_k$.这样全部参数就只有$O(K^2V^2)$个了,就得到了一个紧凑密度模型(compact density model)了(Neal 1992; Frey 1998).如果我们要评估一个全面观测向量$x_{1:t-1}$的概率,这已经足够了.比如可以使用这个模型定义一个类条件密度(class-conditional density)$p(x|y)=c$,然后建立一个生成分类器 (generative classifier,Bengio and Bengio 2000).不过这个模型不能用于其他预测任务,因为这个模型里面每个变量都依赖之前观测的全部变量.所以其他问题要另寻办法.

### 10.1.2 条件独立性(Conditional independence)

高效率表征一个大规模联合分布的关键就是对条件独立性(Conditional independence,缩写为CI)进行假设.回忆本书2.2.4当中,在给定Z的情况下,X和Y的条件独立记作$X \bot Y|Z$,当且仅当条件联合分布可以写成条件边缘分布乘积的时候才成立,如下所示:

$X \bot Y|Z \iff p(X,Y|Z)=p(X|Z)p(Y|Z)$(10.2)

这有啥用呢?假如有$x_{t+1}\bot x_{1:t-1}|x_t$,也就是说在给定当前值的条件下,未来值和过去值独立.这就叫(一阶(first order))马尔科夫假设(Markov assumption).利用这个假设,加上链式规则,就可以写出联合分布形式如下所示:

$p(x_{1:V})=p(x_1)\prod^V_{t=1}p(x_t|x_{t-1})$(10.3)

这就叫做一个(一阶(first order))马尔科夫链(Markov chain).可以通过在状态上的初始分布(initial distribution)$p(x_1=i)$来表示,另外加上一个状态转换矩阵(state transition matrix)$p(x_t=j|x_{t-1}=i)$.更多细节参考本书17.2.

### 10.1.3 图模型

虽然一阶马尔科夫假设对于定义一维序列分布很有用,但对于二维图像.或者三维视频,或者更通用的任意维度的变量集(比如生物通路上的基因归属等等),要怎么定义呢?这时候就需要图模型了.

图模型(Graphical models,缩写为GM)是通过设置条件独立性假设(CI assumption)来表示一个联合分布(joint distribution).具体来说就是图上的节点表示随机变量,而(缺乏的)边缘表示条件独立性假设(CI assumption)(对这类模型的更好命名应该是独立性图(independence diagrams),不过图模型这个叫法已经根深蒂固了.)有几种不同类型的图模型,取决于图是有方向(directed)/无方向(undirected)/或者两者结合.在本章只说有方向图(directed graphs).到第19章再说无方向图(undirected graphs).

此处参考原书图10.1

### 10.1.4 图模型术语(Graph terminology)


在继续讨论之前,先要定义一些基本术语概念,大部分都很好理解.

一个图(graph)$G=(V,\mathcal{E})$包括了一系列节点(node)或者顶点(vertices),$V=\{1,...,V\}$,还有一系列的边(deges)$\mathcal{E} =\{ (s,t):s,t\in V \}$.可以使用邻近矩阵(adjacency matrix)来表示这个图,其中用$G(s,t)$来表示$(s,t)\in \mathcal{E}$,也就是$s\rightarrow t$是凸中的一个遍.如果当且仅当$G(t,s)=1$的时候$G(s,t)=1$,就说这个图是无方向的(undirected),否则就是有方向的(directed).一般假设$G(s,s)=0$,意思是没有自我闭环(self loops).

下面是其他一些要常用到的术语:


* 父节点(Parent)对一个有方向图,一个节点的父节点就是所有节点所在的集合:$pa(s) \overset{\triangle}{=} {t : G(t, s) = 1}$.
* 子节点(Child)对一个有方向图,一个节点的子节点就是从这个节点辐射出去的所有节点的集合:$ch(s) \overset{\triangle}{=} {t : G(s,t) = 1}$.
* 族(Family)对一个有方向图,一个节点的族是该节点以及所有其父节点:$fam(s)= \{s\}\cup pa(s)$.
* 根(root)对一个有方向图,根是无父节点的节点.
* 叶(leaf)对一个有方向图,叶就是无子节点的节点.
* 祖先(Ancestors)对一个有方向图,祖先包括一个节点的父节点/祖父节点等等.也就是说t的祖先是所有通过父子关系向下连接到t的节点:$anc(t)\overset{\triangle}{=} \{s:s\rightsquigarrow t\}$.
* 后代(Descendants)对一个有方向图,后代包括一个节点的子节点/次级子节点等等.也就是s的后代就是可以通过父子关系上溯到s的所有节点集合:$desc(t)\overset{\triangle}{=} \{t:s\rightsquigarrow t\}$.
* 邻节点(Neighbors),对于任意图来说,所有直接连接的节点都叫做邻节点:$nbr(s) \overset{\triangle}{=}  \{t : G(s, t) = 1 \vee G(t, s) = 1\}$.对于无方向图(undirected graph)来说,可以用$s \sim t$来表示s和t是邻节点(这样$(s,t)\in \mathcal{E}$就是图的边(edge)了).












