# MLAPP 读书笔记 - 07 线性回归(Linear regression)

> A Chinese Notes of MLAPP,MLAPP 中文笔记项目 
https://zhuanlan.zhihu.com/python-kivy

记笔记的人：[cycleuser](https://www.zhihu.com/people/cycleuser/activities)

2018年06月15日13:41:04

## 7.1 概论

线性回归是统计学和(监督)学习里面的基本主力.使用核函数或者其他形式基函数来扩展之后,还可以用来对非线性关系进行建模.把高斯输出换成伯努利或者多元伯努利分部,就还可以用到分类上面,这些后文都会讲到.所以这个模型很值得详细学习一下.

## 7.2 模型选择


在本书1.4.5已经看到过线性回归了,其形式为:

$p(y|x\theta)=N(y|w^Tx\sigma^2)$(7.1)

线性回归也可以通过将x替换成为输入特征的非线性函数比如$\phi(x)$来对非线性关系进行建模.也就是将形式变成了:


$p(y|x\theta)=N(y|w^T\phi (x)\sigma^2)$(7.2)


这就叫基函数扩展(basis function expansion).(要注意这时候模型依然是以w为参数,依然还是线性模型;这一点后面会有很大用处.)简单的粒子就是多项式基函数,模型中函数形式为:

$\phi(x)=[1,x,x^2,...,x^d]$(7.3)

图1.18展示了改变d的效果,增加d就可以建立更复杂的函数.

对于多输入的模型,也可以使用线性回归.比如将温度作为地理位置的函数来建模.图7.1(a)所示为:$\mathrm{E}[y|x]=w_0+w_1x_1+w_2x_2$,图7.1(b)所示为:$\mathrm{E}[y|x]=w_0+w_1x_1+w_2x_2+w_3x_1^2+w_4x_2^2$.


## 7.3 最大似然估计(最小二乘法)

最大似然估计(MLE)是估计统计模型参数的常用方法了,定义如下:

$ \hat\theta \overset{\triangle}{=} \arg\max_\theta \log p(D|\theta)$(7.4)

此处参考原书图7.1

通常假设训练样本都是独立同分布的(independent and identically distributed,缩写为iid).这就意味着可以写出下面的对数似然函数(log likelihood):

$l(\theta) \overset{\triangle}{=} \sum^N_{i=1}\log p(y_i|x_i,\theta)$(7.5)

我们可以去最大化对数似然函数,或者也可以等价的最小化负数对数似然函数(the negative log likelihood,缩写为NLL):

$NLL(\theta)\overset{\triangle}{=} -\sum^N_{i=1}\log p(y_i|x_i,\theta)$(7.6)

负对数似然函数(NLL)有时候更方便,因为很多软件都有专门设计找最小值的函数,所以比最大化容易.

接下来设我们对这个线性回归模型使用最大似然估计(MLE)方法.在上面的公式中加入高斯分布的定义,就得到了下面形式的对数似然函数:

$$
\begin{aligned}
l(\theta)&=  \sum^N_{i=1}\log[(\frac{1}{2\pi\sigma^2})^{\frac{1}{2}} \exp (-\frac{1}{2\sigma^2}(y_i-w^Tx_i)^2)   ]&\text{(7.7)}\\
&= \frac{-1}{2\sigma^2}RSS(w)-\frac{N}{2}\log(2\pi\sigma^2) &\text{(7.8)}\\
\end{aligned}
$$

上式中的RSS是residual sum of squares的缩写,意思是残差平方定义为:
$RSS(w)\overset{\triangle}{=} \sum^N_{i=1}(y_i-w^Tx_i)^2$    (7.9)


此处参考原书图7.2

RSS也叫做平方误差总和(sum of squared errors),这样也可以缩写成SSE,这样就有SSE/N,表示的是均方误差MSE(mean squared erro).也可以写成残差(residual errors)向量的二阶范数(l2 norm)的平方和:



$RSS(w)=||\epsilon||^2_2=\sum^N_{i=1}\epsilon_i^2$     (7.10)


上式中的$\epsilon_i=(y_i-w^Tx_i)^2$.

这样就能发现w的最大似然估计(MLE)就是能让残差平方和(RSS)最小的w,所以这个方法也叫作小二乘法(least squares).这个方法如图7.2所示.图中红色圆点是训练数据$x_i,y_i$,蓝色的十字点是估计数据$x_i,\hat y_i$,竖直的蓝色线段标识的就是残差$\epsilon_i=y_i-\hat y_i$.目标就是要寻找能够使平方残差总和(图中蓝色线段长度)最小的图中所示红色直线的参数(斜率$w_1$和截距$w_0$).

在图7.2(b)中是线性回归样例的负对数似然函数(NLL)曲面.可见其形态类似于一个单底最小值的二次型碗,接下来就要进行以下推导.(即便使用了基函数扩展,比如多项式之类的,这也是成立的,因为虽然输入特征可以不是线性的,单负对数似然函数依然还是以w为参数的线性函数.)


### 7.3.1 最大似然估计(MLE)的推导

首先以更好区分的形式重写目标函数(负对数似然函数):

$NLL(w)=\frac{1}{2}(y-Xw)^T(y-Xw)=\frac{1}{2}w^T(X^TX)w-w^T(X^Ty)$   (7.11)

上式中
$X^TX=\sum^N_{i=1}x_ix_u^T=\sum^N_{i=1}\begin{pmatrix} x_{i,1}^2&... x_{i,1}x_{i,D}\\&& ...&\\  x_{i,D}x_{i,1} &... & x_{i,D}^2 \end{pmatrix}$   (7.12)

是矩阵平方和（sum of squares matrix）,另外的一项为：

$X^Ty=\sum^N_{i=1}x_iy_i$   (7.13)

使用等式4.10中的结论,就得到了梯度函数（gradient）,如下所示：

$g(w)=[X^TXw-X^Ty]=\sum^N_{i=1} x_i(w^Tx_i-y_i)$   (7.14)

使梯度为零,则得到了：

$X^TXw=X^Ty$   (7.15)


这就是正规方程（normal equation）.这个线性方程组对应的解$\hat w$就叫做常规最小二乘解（ordinary least squares solution,缩写为 OLS solution）：

$\hat w_{OLS}=(X^TX)^{-1}X^Ty$   (7.16)重要公式



### 7.3.2 几何解释


这个方程有很优雅的几何解释.假设N>D,也就意味样本比特征数目多.X列向量（columns）定义的是在N维度内的一个D维度的子空间.设第j列为$\tilde x_j$,是在$R^N$上的一个向量.(应该不难理解,$x_i\in R^D$表示的就是数据情况中的第i个.)类似的y也是一个$R^N$中的向量.例如,如果社N=3个样本,二D=2的子空间:

$X=\begin{pmatrix}1&2 \\ 1 &-2\\1 &2 \end{pmatrix},y=\begin{pmatrix}8.8957\\0.6130\\1.7761\end{pmatrix}$   (7.17)

这两个向量如图7.3所示.

然后我们就要在这个线性子空间中找一个尽可能靠近y的向量$\hat y\in R^N$,也就是要找到:

$\arg\min_{\hat\in span(\{ \tilde x_1,...,\tilde x_D \})} ||y-\hat y||_2$   (7.18)

由于$\hat y \in span(X)$,所以就会存在某个权重向量(weight vector)w使得:

$\hat y= w_1\tilde x_1+...+w_D\tilde x_D=Xw$   (7.19)


此处参考原书图7.3


要最小化残差的范数(norm of the residual)$y-\hat y$,就需要让残差向量(residual vector)和X的每一列相正交(orthogonal),也就是对于$j=1:D$有$\tilde x ^T_j (y-\hat y) =0$.因此有:



$\tilde x_j^T(y-\hat y)=0  \implies X^T(y-Xw)=0\implies w=(X^TX)^{-1}X^Ty$ (7.20)

这样y的投影值就是:

$\hat y=X\hat w= X(X^TX)^{-1}X^Ty$(7.21)

这对应着在X的列空间(column space)中的y的正交投影(orthogonal projection).投影矩阵$P\overset{\triangle}{=} X(X^TX)^{-1}X^T$就叫做帽子矩阵(hat matrix),因为在y上面盖了个帽子成了$\hat y$.


### 7.3.3 凸性质

在讲到最小二乘法的时候,我们注意到负对数似然函数(NLL)形状像是一个碗,有单一的最小值.这样的函数用专业术语来说是凸(convex)的函数.凸函数在机器学习里面非常重要.

然后咱们对这个概念进行一下更确切定义.设一个集合S,如果对于任意的$\theta,\theta'\in S$,如果有下面的性质,则S是凸的集合:

$\lambda\theta+(1-\lambda)\theta'\in S, \forall  \lambda\in[0,1]$(7.22)

此处参考原书图7.4
此处参考原书图7.5

也就是说在$\theta$和$\theta'$之间连一条线,线上所有的点都处在这个集合之内.如图7.4(a)所示就是一个凸集合,而图7.4(b)当中的就是一个非凸集合.

一个函数的上图(epigraph,也就是一个函数上方的全部点的集合)定义了一个凸集合,则称这个函数$f(\theta)$就是凸函数.反过来说,如果定义在一个凸集合上的函数$f(\theta)$满足对任意的$\theta,\theta'\in S$,以及任意的$0\le\lambda\le1$,都有下面的性质,也说明这个函数是凸函数:

$f(\lambda \theta +(1-\lambda)\theta')\le \lambda f(\theta) +(1-\lambda)f(\theta ')$ (7.23)

图7.5(b)是一个一维样本.如果不等式严格成立,就说这个函数是严格凸函数(strictly convex).如果其反函数$-f(\theta)$是凸函数,则这个函数$f(\theta$是凹函数(concave).标量凸函数(scalar convex function)包括$\theta^2,e^\theta,\theta\log\theta (\theta>0)$.标量凹函数(scalar concave function)包括$\log(\theta),\sqrt\theta$.

直观来看,(严格)凸函数就像是个碗的形状,所以对应在碗底位置有全局的唯一最小值$\theta^*$.因此其二阶导数必须是全局为正,即$\frac{d}{d\theta}f(\theta)>0$.当且仅当一个二阶连续可微(twice-continuously differentiable)多元函数f的海森矩阵(Hessian)对于所有的$\theta$都是正定的(positive definite),这个函数才是凸函数.在机器学习语境中,这个函数f通常对应的都是负对数似然函数(NLL).

此处参考原书图7.6


负对数似然函数(NLL)是凸函数的模型是比较理想的.因为这就意味着能够找到全局最优的最大似然估计(MLE).本书后面还会看到很多这类例子.不过很多模型还并不一定就能有凹的似然函数.这时候就要推一些方法来求局部最优参数估计了.









