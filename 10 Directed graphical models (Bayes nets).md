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

图模型(Graphical models,缩写为GM)是通过设置条件独立性假设(CI assumption)来表示一个联合分布(joint distribution).具体来说就是图上的节点表示随机变量,而(缺乏的)边缘表示条件独立性假设(CI assumption)(对这类模型的更好命名应该是独立性图(independence diagrams),不过图模型这个叫法已经根深蒂固了.)有几种不同类型的图模型,取决于图是有向(directed)/无向(undirected)/或者两者结合.在本章只说有向图(directed graphs).到第19章再说无向图(undirected graphs).

此处参考原书图10.1

### 10.1.4 图模型术语(Graph terminology)


在继续讨论之前,先要定义一些基本术语概念,大部分都很好理解.

一个图(graph)$G=(V,\mathcal{E})$包括了一系列节点(node)或者顶点(vertices),$V=\{1,...,V\}$,还有一系列的边(deges)$\mathcal{E} =\{ (s,t):s,t\in V \}$.可以使用邻近矩阵(adjacency matrix)来表示这个图,其中用$G(s,t)$来表示$(s,t)\in \mathcal{E}$,也就是$s\rightarrow t$是凸中的一个遍.如果当且仅当$G(t,s)=1$的时候$G(s,t)=1$,就说这个图是无向的(undirected),否则就是有向的(directed).一般假设$G(s,s)=0$,意思是没有自我闭环(self loops).

下面是其他一些要常用到的术语:


* 父节点(Parent)对一个有向图,一个节点的父节点就是所有节点所在的集合:$pa(s) \overset{\triangle}{=} {t : G(t, s) = 1}$.
* 子节点(Child)对一个有向图,一个节点的子节点就是从这个节点辐射出去的所有节点的集合:$ch(s) \overset{\triangle}{=} {t : G(s,t) = 1}$.
* 族(Family)对一个有向图,一个节点的族是该节点以及所有其父节点:$fam(s)= \{s\}\cup pa(s)$.
* 根(root)对一个有向图,根是无父节点的节点.
* 叶(leaf)对一个有向图,叶就是无子节点的节点.
* 祖先(Ancestors)对一个有向图,祖先包括一个节点的父节点/祖父节点等等.也就是说t的祖先是所有通过父子关系向下连接到t的节点:$anc(t)\overset{\triangle}{=} \{s:s\rightsquigarrow t\}$.
* 后代(Descendants)对一个有向图,后代包括一个节点的子节点/次级子节点等等.也就是s的后代就是可以通过父子关系上溯到s的所有节点集合:$desc(t)\overset{\triangle}{=} \{t:s\rightsquigarrow t\}$.
* 邻节点(Neighbors),对于任意图来说,所有直接连接的节点都叫做邻节点:$nbr(s) \overset{\triangle}{=}  \{t : G(s, t) = 1 \vee G(t, s) = 1\}$.对于无向图(undirected graph)来说,可以用$s \sim t$来表示s和t是邻节点(这样$(s,t)\in \mathcal{E}$就是图的边(edge)了).
* 度数(Degree)一个节点的度数是指该节点的邻节点个数.对有向图,又分为入度数(in-degree)和出度数(out-degree),指代的分别是某个节点的父节点和子节点的个数.
* 闭环(cycle/loop)顾名思义,只要沿着一系列节点能回到初始的位置,顺序为$s_1 \rightarrow s_2 ... \rightarrow s_n \rightarrow s_1, n \ge 2$,就称之为一个闭环.如果图是有向的,闭环也是有向的.图10.1(a)中没有有向的闭环,倒是有个无向闭环$1\rightarrow 2\rightarrow 4 \rightarrow 3 \rightarrow 1$.
* 有向无环图(directed acyclic graph,缩写为DAG)顾名思义,就是有向但没有有向闭环的,比如图10.1(a)就是一例.
* 拓扑排序(Topological ordering)对一个有向无环图(DAG),拓扑排序(topological ordering)或者也叫全排序(total ordering)是所有父节点比子节点数目少的节点的计数.比如在图10.1(a)中,就可以使用(1, 2, 3, 4, 5)或者(1, 3, 2, 5, 4)等.
* 路径(Path/trail)对于$s\rightsquigarrow t$来说路径就是一系列从s到t的有向边(directed edges).
* 树(Tree)无向树(undirected tree)就是没有闭环的无向图.有向树(directed tree)是没有有向闭环的有向无环图(DAG).如果一个节点可以有多个父节点,就称为超树(polytree),如果不能有多个父节点,就成为规范有向树(moral directed tree).
* 森林(Forest)就是树的集合.
* 子图(Subgraph)(包括节点的)子图$G_A$是使用A中的节点和对应的边(edges)创建的$G_A=(\mathcal{V}_A,\mathcal{E}_A)$.
* 团(clique)对一个无向图,团是一系列互为邻节点的节点的集合.在不损失团性质的情况下能达到的最大规模的团就叫做最大团(maximal clique).比如图10.1(b)
当中的{1,2}就是一个团,但不是最大团,因为把3加进去依然保持了团性质(clique property).图10.1(b)中的最大团:{1, 2, 3}, {2, 3, 4}, {3, 5}.

### 10.1.5 有向图模型

有向图模型(directed graphical model,缩写为DGM)是指整个图都是有向无环图(directed acyclic graph,缩写为DAG)的图模型.更广为人知的名字叫贝叶斯网络(Bayesian networks).不过实际上这个名字并不是说这个模型和贝叶斯方法有啥本质上的联系:只是定义概率分布的一种方法而已.这些模型也叫作信念网络(belief networks).这里的信念这个词(belief)指的是主观的概率.关于表征有向图模型(DGM)的概率分布的种类并没有什么本质上的主管判断.这些模型有时候也叫作因果网络(causal networks),因为有时候可以将有向箭头解释成因果关系.不过有向图模型本质上并没有因果关系(关于因果关系有向图模型的讨论参考本书26.6.1.)名字这么多这么乱,咱们就选择最中性的称呼,就叫它有向图模型(DGM).

有向无环图(DAG)的关键性之就是节点可以按照父节点在子节点之前来排序.这也叫做拓扑排序(topological ordering),在任何有向无环图中都可以建立这种排序.给定一个这样的排序,就定义了一个有序马尔科夫性质(ordered Markov property),也就是假设一个节点只取决于其直接父节点,而不受更早先辈节点的影响.也就是:

$x_s \bot  x_{pred(s) / pa(s)}| x_{pa(s)}$(10.4)

(注:上面公式中应该是反斜杠 "\",但是我不知在LaTex里面怎么打出来.)

上式中的$pa(s)$表示的是节点s的父节点,而$pred(s)$表示在排序中s节点的先辈节点.这是对一阶马尔科夫性质(first-order Markov property)的自然扩展,目的是构成一个链条来泛华有向无环图(DAG).

此处参考原书图10.2

例如,在图10.1(a)中编码了下面的联合分布:

$$
\begin{aligned}
p(x_{1:5}) &= p(x_1)p(x_2|x_1)p(x_3|x_1,\cancel{x_2})p(x_4|\cancel{x_1}, x_2, x_3)p(x_5|\cancel{x_1},\cancel{x_2},x_3,\cancel{x_4}) &\text{(10.5)}\\
&=  p(x_1)p(x_2|x_1)p(x_3|x_1)p(x_4|x_2, x_3)p(x_5|x_3)   &\text{(10.6)}
\end{aligned}
$$

其中每个$p(x_t|x_{pa(t)})$都是一个条件概率分布(CPD).将分布写成$p(x|G)$是要强调只有当有向无环图(DAG)G当中编码的条件独立性假设(CI assumption)是正确的时候登时才能成立.不过通常为了简单起见都会把这个条件忽略掉.每个节点都有$O(F)$个父节点以及K个状态,这个模型中参数总数就是$O(VK^F)$,这就比不做独立性假设的模型所需要的$O(K^V)$个要少多了.

## 10.2 样例

这一节展示一些可以用有向图模型(DGM)来表示的常见概率模型.

### 10.2.1 朴素贝叶斯分类器(Naive Bayes classifiers)

朴素贝叶斯分类器是在本书3.5就讲到了.这个模型中假设了给定类标签(calss label)条件下特征条件独立.此假设如图10.2(a)所示.这样就可以将联合分布写成下面的形式:
$p(y,x)=p(y)\prod^D_{j=1}p(x-j|y)$(10.8)

朴素贝叶斯加设很朴素,因为假设了所有特征都是条件独立的.使用图模型就是捕获变量之间相关性的一种方法.如果模型是属性的,就称此方法为树增强朴素贝叶斯分类器(tree-augmented naive Bayes classifiers,缩写为TAN,Friedman et al. 1997).如图10.2(b)所示.使用树形结构而不是一般的图模型的原因有两方面.首先是树形结构使用朱刘算法(Chow-Liu algorithm,1965年有朱永津和刘振宏提出)来优化,具体如本书26.3所示.另外就是树结构模型的缺失特征好处理,具体会在本书20.2有解释.

此处参考原书图10.3


此处参考原书图10.4

### 10.2.2 马尔科夫和隐形马尔科夫模型






