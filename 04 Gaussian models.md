# MLAPP 读书笔记 - 04 高斯模型(Gaussian models)

> A Chinese Notes of MLAPP，MLAPP 中文笔记项目 
https://zhuanlan.zhihu.com/python-kivy

记笔记的人：[cycleuser](https://www.zhihu.com/people/cycleuser/activities)

2018年05月16日10:49:49


## 4.1 简介

本章要讲的是多元高斯分布(multivariate Gaussian),或者多元正态分布(multivariate normal ,缩写为MVN)模型,这个分布是对于连续变量的联合概率密度函数建模来说最广泛的模型了.未来要学习的其他很多模型也都是以此为基础的.

然而很不幸的是,本章所要求的数学水平也是比很多其他章节都要高的.具体来说是严重依赖线性代数和矩阵积分.要应对高维数据,这是必须付出的代价.初学者可以跳过标记了星号的章.另外本章有很多等式,其中特别重要的用方框框了起来.

### 4.1.1 记号

这里先说几句关于记号的问题.向量用小写字母粗体表示,比如**x**.矩阵用大写字母粗体表示,比如**X**.大写字母加下标表示矩阵中的项,比如$X_{ij}$.
所有向量都假设为列向量(column vector),除非特别说明是行向量.通过堆叠(stack)D个标量(scalar)得到的类向量记作$[x_1,...,x_D]$.与之类似,如果写**x=**$[x_1,...,x_D]$,那么等号左侧就是一个高列向量(tall column vector),意思就是沿行堆叠$x_i$,一般写作**x=**$(x_1^T,...,x_D^T)^T$,不过这样很丑哈.如果写**X=**$[x_1,...,x_D]$,等号左边的是矩阵,意思就是沿列堆叠$x_i$,建立一个矩阵.

### 4.1.2 基础知识

回想一下本书2.5.2中关于D维度下的多元正态分布(MVN)概率密度函数(pdf)的定义,如下所示:
$N(x|\mu,\Sigma)*= \frac{1}{(2\pi)^{D/2}|\Sigma |^{1/2}}\exp[ -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)]$(4.1 重要公式)


此处参考原书图4.1

指数函数内部的是一个数据向量**x**和均值向量**$\mu$** 之间的马氏距离(马哈拉诺比斯距离,Mahalanobis distance).对$\Sigma$进行特征分解(eigendecomposition)有助于更好理解这个量.$\Sigma = U\wedge U ^T$,其中的U是标准正交矩阵(orthonormal matrix),满足$U^T U = I$,而$\wedge $是特征值组成的对角矩阵.经过特征分解就得到了:

$\Sigma^{-1}=U^{-T}\wedge^{-1}U^{-1}=U\wedge ^{-1}U^T=\sum^D_{i=1}\frac{1}{\lambda_i}u_iu_i^T$(4.2)

上式中的$u_i$是U的第i列,包含了第i个特征向量(eigenvector).因此就可以把马氏距离写作:


$$\begin{aligned}
(x-\mu)^T\Sigma^{-1}(x-\mu)&=(x-\mu)^T(\sum^D_{i=1}\frac{1}{\lambda_i}u_iu_i^T)(x-\mu)&\text{(4.3)}\\
&= \sum^D_{i=1}\frac{1}{\lambda_i}(x-\mu)^Tu_iu_i^T(x-\mu)=\sum^D_{i=1}\frac{y_i^2}{\lambda_i}&\text{(4.4)}\\
\end{aligned}$$

上式中的$y_i*= u_i^T(x-\mu)$.二维椭圆方程为:
$\frac{y_1^2}{\lambda_1}+\frac{y_2^2}{\lambda_2}=1$(4.5)

因此可以发现高斯分布的概率密度的等值线沿着椭圆形,如图4.1所示.特征向量决定了椭圆的方向,特征值决定了椭圆的形态即宽窄比.

一般来说我们将马氏距离(Mahalanobis distance)看作是对应着变换后坐标系中的欧氏距离(Euclidean distance),平移$\mu$,旋转U.

### 4.1.3 多元正态分布(MVN)的最大似然估计(MLE)

接下来说的是使用最大似然估计(MLE)来估计多元正态分布(MVN)的参数.在后面的章节里面还会说道用贝叶斯推断来估计参数,能够减轻过拟合,并且能对估计值的置信度提供度量.

#### 定理4.1.1(MVN的MLE)
如果有N个独立同分布样本符合正态分布,即$x_i ∼ N(\mu,\Sigma)$,则对参数的最大似然估计为:
$\hat\mu_{mle}=\frac{1}{N}\sum^N_{i=1}x_i *= \bar x$(4.6)
$\hat\Sigma_{mle}=\frac{1}{N}\sum^N_{i=1}(x_i-\bar x)(x_i-\bar x)^T=\frac{1}{N}(\sum^N_{i=1}x_ix_i^T)-\bar x\bar x^T$(4.7)

也就是MLE就是经验均值(empirical mean)和经验协方差(empirical covariance).在单变量情况下结果就很熟悉了:
$\hat\mu =\frac{1}{N}\sum_ix_i=\bar x $(4.8)
$\hat\sigma^2 =\frac{1}{N}\sum_i(x_i-x)^2=(\frac{1}{N}\sum_ix_i^2)-\bar x^2$(4.9)

#### 4.1.3.1 证明

要证明上面的结果,需要一些矩阵代数的计算,这里总结一下.等式里面的**a**和**b**都是向量,**A**和**B**都是矩阵.记号$tr(A)$表示的是矩阵的迹(trace),是其对角项求和,即$tr(A)=\sum_i A_{ii}$.

$$
\begin{aligned}
\frac{\partial(b^Ta)}{\partial a}&=b\\
\frac{\partial(a^TAa)}{\partial a}&=(A+A^T)a\\
\frac{\partial}{\partial A} tr(BA)&=B^T\\
\frac{\partial}{\partial A} \log|A|&=A^{-T}*= (A^{-1})^T\\
tr(ABC)=tr(CAB)&=tr(BCA)
\end{aligned}
$$(4.10 重要公式)


上式中最后一个等式也叫做迹运算的循环置换属性(cyclic permutation property).利用这个性质可以推广出很多广泛应用的求迹运算技巧,对标量内积$x^TAx$就可以按照如下方式重新排序:
$x^TAx=tr(x^TAx)=tr(xx^TA)=tr(Axx^T)$(4.11)


##### 证明过程
接下来要开始证明了,对数似然函数为:

$l(\mu,\Sigma)=\log p(D|\mu,\Sigma)=\frac{N}{2}\log|\wedge| -\frac{1}{2}\sum^N_{i=1}(x_i-\mu)^T\wedge (x_i-\mu) $(4.12)

上式中$\wedge=\Sigma^{-1}$,是精度矩阵(precision matrix)

