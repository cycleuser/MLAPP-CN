# MLAPP 读书笔记 - 09 通用线性模型(Generalized linear models)和指数族分布(exponential family)

> A Chinese Notes of MLAPP,MLAPP 中文笔记项目 
https://zhuanlan.zhihu.com/python-kivy

记笔记的人：[cycleuser](https://www.zhihu.com/people/cycleuser/activities)

2018年6月27日09:26:15

## 9.1 概论

之前已经见到过很多概率分布了:正态(高斯)分布,伯努利分布(Bernoulli),学生T分布,均匀分布,$\gamma$分布等等.这些大多数都属于指数族分布(exponential family).本章就要讲这类分布的各种特点.然后我们就能用来推出很多广泛应用的定力和算法.

接下来我们会看到要构建一个生成分类器如何简单地用指数族分布中的某个成员来作为类条件密度.另外还会讲到如何构建判别模型,其中响应变量服从指数族分布,均值是输入特征的线性函数;这就叫做通用线性模型(Generalized linear models),将逻辑回归上的思想扩展到了其他类别的响应变量上去.


## 9.2 指数族分布

在定义指数族分布之前,先要说一下这东西重要的几个原因:
* 在特定的规范化条件下(regularity conditions),指数族分布是唯一有限规模充分统计量(finite-sized sufficient statistics)的分布族,这意味着可以将数据压缩称固定规模的浓缩概括而不损失信息.这在在线学习情况下特别有用,后面会看到.
* 指数族分布是唯一有共轭先验的分布族,这就简化了后验的计算,参考本书9.2.5.
* 指数族分布对于用户选择的某些约束下有最小假设集合,参考本书9.2.6.
* 指数族分布是通用线性模型(generalized linear models)的核心,参考本书9.3.
* 指数族分布也是变分推理(variational inference)的核心,参考本书21.2.

### 9.2.1 定义

概率密度函数(pdf)或者概率质量函数(pmf)$p(x|\theta)$,对$x=(x_1,...,x_m)\in X^m, \theta\in \Theta \subseteq R^d$,如果满足下面的形式,就称其属于指数族分布(exponential family):

$$
\begin{aligned}
p(x|\theta) &= \frac{1}{Z(\theta)} h(x) \exp[\theta^T\phi(x)] &\text{(9.1)}\\
&=  h(x) \exp[\theta^T\phi(x)-A(\theta)] &\text{(9.2)}\\
\end{aligned}
$$

其中:
$$
\begin{aligned}
Z(\theta) &= \int_{X_m}  h(x) \exp[\theta^T\phi(x)] dx &\text{(9.3)}\\
A(\theta) &=  \log Z(\theta) &\text{(9.4)}\\
\end{aligned}
$$

因此$\theta$也叫作自然参数(natural parameters)或者规范参数(canonical parameters),$\phi(x)\in R^d$叫做充分统计向量(vector of sufficient statistics),$Z(\theta)$叫做分区函数(partition function),$A(\theta)$叫做对数分区函数(log partition function)或者累积函数(cumulant function),$h(x)$是一个缩放常数,通常为1.如果$\phi(x)=x$,就说这个是一个自然指数族(natural exponential family).

等式9.2可以泛华扩展,写成下面这种方式:
$p(x|\theta)=h(x)\exp[\eta(\theta)^T\phi(x)-A(\eta(\theta))]$(9.5)

其中的$\eta$是一个函数,将参数$\theta$映射到规范参数(canonical parameters)$\eta= \eta(\theta)$,如果$dim(\theta)<dim(\eta(\theta))$,就成了弯曲指数族(curved exponential family),意味着充分统计(sufficient statistics)比参数(parameters)更多.如果$\eta(\theta)=\theta$,就说这个模型是规范形式的(canonical form).如果不加额外声明,都默认假设模型都是规范形式的.

### 9.2.2 举例

接下来举几个例子以便理解.

#### 9.2.2.1 伯努利分布(Bernoulli)


$x\in \{0,1\}$的伯努利分布写成指数族形式如下所示:
$Ber(x|\mu) =\mu^x(1-\mu)^{1-x} =\exp[x\log(\mu)+(1-x)\log(1-\mu)] =\exp[\phi(x)^T\theta]$(9.6)


其中的$\phi(x)=[I(x0),I(x=1),\theta =[\log(\mu)+(1-x)\log(1-\mu)]]$.不过这种表示是过完备的(over-complete),因为在特征(features)之间有一个线性依赖关系:

$1^T\[hi(x)=I(x=0)+I(x=1)=1$(9.7)

结果$\theta$就不再是唯一可识别的(uniquely identifiable).通常都要求表述最简化(minimal),也就意味着关于这个分布要有为一个的$\theta$.可以定义如下:

$Ber(x|\mu)=(1-\mu)\exp[x\log(\frac{\mu}{1-\mu})]$(9.8)

现在就有了$\phi(x)=x,\log(\frac{\mu}{1-\mu}$就是对数比值比(log-odds ratio),$Z=1/(1-\,u)$.然后可以从规范参数里面恢复出均值参数$\mu$:

$\mu=sigm(\theta)=\frac{1}{1+e^{-\theta}}$(9.9)


#### 9.2.2.2 多重伯努利(Multinoulli)

多重伯努利分布表述成最小指数族如下所示(其中的$x_k\I(x=k)$):
$$
\begin{aligned}
Cat(x|\mu)&= \prod^K_{k=1} \mu_k^{x_k} =\exp[\sum^K_{k=1}x_k\log \mu_k]     &\text{(9.10)}\\
&=  \exp[\sum^{K-1}_{k=1}x_k\log \mu_k   +(1- \sum^{K-1}_{k=1}x_k)\log(1- \sum^{K-1}_{k=1}\mu_k)  ]     &\text{(9.11)}\\
&=  \exp[  \sum^{K-1}_{k=1}x_k \log(\frac{\mu_k}{1-\sum^{K-1}_{j=1}\mu_j}) +\log(1-\sum^{K-1}_{k=1}\mu_k) ]     &\text{(9.12)}\\
&= \exp[\sum^{K-1}_{k=1}x_k\log( \frac{\mu_k}{\mu_K})+\log \mu_k]     &\text{(9.13)}\\
\end{aligned}
$$
其中$\mu_K=1-\sum^{K-1}_{k=1}\mu_k$.也可以写成下面的指数族形式

$$
\begin{aligned}
Cat(x|\theta)&= \exp(\theta^T\phi(x)-A(\theta))      &\text{(9.14)}\\
\theta &= [\log \frac{\mu_1}{\mu_K},...,\log \frac{\mu_{K-1}}{\mu_K}]     &\text{(9.15)}\\
\phi(x) &=  [I(x=1),...,I(x=K-1)]    &\text{(9.16)}\\
\end{aligned}
$$

从规范参数里面恢复出均值参数$\mu$:

$\mu_k =\frac{e^{\theta_k}{1+\sum^{K-1}_{j=1} e^{\theta_j}}$(9.17)

然后就能发现:

$\mu_K= 1-\frac{\sum^{K-1}_{j=1} e^{\theta_j}}{1+\sum^{K-1}_{j=1} e^{\theta_j}}=\frac{1}{\sum^{K-1}_{j=1} e^{\theta_j}}$(9.18)

因此:

$A(\theta)=\log(1+\sum^{K-1}_{k=1} e^{\theta_k})$(9.19)


如果定义$\theta_K=0$,就可以卸除$\mu=S(\theta),A(\theta)= \log\sum^K_{k=1}e^{\theta_k}$,其中的S是等式4.39中的柔性最大函数(softmax function).

#### 9.2.2.3 单变量高斯分布(Univariate Gaussians)


单变量高斯分布写成指数族形式如下所示:

$$
\begin{aligned}
  N(x|\mu,\sigma^2) &= \frac{1}{(2\pi\sigma^2)^{1/2}} \exp[-\frac{1}{2\sigma^2}(x-\mu)^2]    &\text{(9.20)}\\
  &=  \frac{1}{(2\pi\sigma^2)^{1/2}} \exp[-\frac{1}{2\sigma^2}x^2 +\frac{\mu}{\sigma^2}x  -\frac{1}{2\sigma^2}\mu^2]   &\text{(9.21)}\\
  &= \frac{1}{Z(\theta}\exp(\theta^T\phi(x)))}   &\text{(9.22)}\\
\end{aligned}
$$

其中
$$
\begin{aligned}
 \theta &= \begin{pmatrix} \mu/\sigma^2 \\\frac{-1}{2\sigma^2}  \end{pmatrix}   &\text{(9.23)}\\
 \phi(x) &= \begin{pmatrix} x\\x^2   \end{pmatrix}      &\text{(9.24)}\\
 Z(\mu,\sigma^2) &= \sqrt{2\pi}\sigma \exp [\frac{\mu^2}{2\sigma^2}]   &\text{(9.25)}\\
 A(\theta) &= \frac{-\theta_1^2}{4\theta_2}-\frac{1}{2}\log(-2\theta_2)-\frac{1}{2} \log(2\pi)   &\text{(9.26)}\\
\end{aligned}
$$


