# MLAPP 读书笔记 - 08 逻辑回归回归(Logistic regression)

> A Chinese Notes of MLAPP,MLAPP 中文笔记项目 
https://zhuanlan.zhihu.com/python-kivy

记笔记的人：[cycleuser](https://www.zhihu.com/people/cycleuser/activities)

2018年6月20日15:05:14

## 8.1 概论

构建概率分类器有一种方法是建立形式为$p(y,x)$的联合模型,然后以x为条件,推$p(y|x)$.这叫生成方法(generative approach).另外一种办法是直接以$p(y|x)$的形式去拟合一个模型.这就叫做启发式方法(discriminative approach),本章就要讲这个方法.具体来说就是要假设有一些参数为线性的启发模型.我们会发现这样的模型拟合起来特别简单.在8.6,我们会对生成方法和启发方法进行对比,在本书后面的章节中,还会讲到非线性的启发模型和非参数化的启发模型.

## 8.2 模型选择

正如在本书1.4.6讲过的,逻辑回归对应的是下面这种二值化分类模型:
$p(y|x,w)=Ber(y|sigm(w^Tx))$(8.1)

图1.19(b)所示的就是一个一维例子.逻辑回归可以很容易扩展用于高维度输入特征向量.比如图8.1所示的就是对二维输入和不同的权重向量w的逻辑回归$p(y=1|x,w)=sigm(w^Tx)$.如果设置概率0.5的位置为阈值,就能得到一个决策边界,其范数(垂线)为w.

## 8.3 模型拟合

本节讲对逻辑回归模型进行参数估计的算法.

此处参考原书图8.1

### 8.3.1 最大似然估计(MLE)

逻辑回归的负对数似然函数为:
$$
\begin{aligned}
NLL(w)&= -\sum^N_{i=1}\log [\mu_i^{I(y_i=1)}\times (1-\mu_i)^{I(y_i=0)}]  &\text{(8.2)}\\
&=  -\sum^N_{i=1}\log [y_i\log \mu_i+(1-y_i)\log(1-\mu_i)]   &\text{(8.3)}\\
\end{aligned}
$$

这也叫交叉熵误差函数(cross-entropy error function)参考本书2.8.2.

这个公式还有另外一个写法,如下所示.设$\tilde y_i \in \{-1,+1\}$,而不是$y_i\in\{0,1\}$.另外还有$p(y=-1)=\frac{1}{ 1+\exp (-w^Tx)}$和$p(y=1)=\frac{1}{ 1+\exp (+w^Tx)}$.这样有:

$NLL(w)=\sum^N_{i=1}\log(1+\exp(-\tilde y_i w^Tx_i))$(8.4)

和线性回归不一样的是,在逻辑回归里面,我们不再能以闭合形式写出最大似然估计(MLE).所以需要使用一个优化算法来计算出来.为了这个目的,就要推到梯度(gradient)和海森矩阵(Hessian).

此处参考原书图8.2

如练习8.3所示,很明显梯度和海森矩阵分别如下所示:
$$
\begin{aligned}
g &=\frac{d}{dw}f(w)=\sum_i*\mu_i-y_i)x_i=X^T(\mu-y)   &\text{(8.5)}\\
H &= \frac{d}{dw}g(w)^T=\sum_i(\nabla_w\mu_i)x_i^T=\sum_i\mu_i(1-\mu_i)x_xx_i^T  &\text{(8.6)}\\
&= X^TSX  &\text{(8.7)}\\
\end{aligned}
$$

其中的$S\overset{\triangle}{=}diag(\mu_i(1-\mu_i))$.通过练习8.3也可以证明海森矩阵H是正定的(positive definite).因此负对数似然函数NLL就是凸函数(convex)有唯一的全局最小值.接下来就说一下找到这个最小值的方法.
