# MLAPP 读书笔记 - 07 线性回归(Linear regression)

> A Chinese Notes of MLAPP，MLAPP 中文笔记项目 
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