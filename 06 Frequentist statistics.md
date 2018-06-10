# MLAPP 读书笔记 - 06 频率论统计(Frequentist statistics)

> A Chinese Notes of MLAPP，MLAPP 中文笔记项目 
https://zhuanlan.zhihu.com/python-kivy

记笔记的人：[cycleuser](https://www.zhihu.com/people/cycleuser/activities)

2018年06月09日10:03:31

## 6.1 概论

第五章中讲的都是贝叶斯统计学(Bayesian statistics).贝叶斯统计学被一些人认为有争议,不过在非统计学领域,贝叶斯统计的应用却没什么争议,比如医疗诊断(本书2.2.3.1)/垃圾邮件过滤(本书3.4.4.1)/飞机追踪(本书18.2.1)等.反对者的理由与统计模型参数和其他未知量之间的区别有关.

然后就有人做出尝试,去避免把参数当作随机变量来推导统计学方法,这样就不需要使用先验和贝叶斯规则了.这种统计学就是频率论统计学(frequentist statistics),也叫经典统计学(classical statistics)或者正统统计学(orthodox statistics).这种统计学不是基于后验分布(posterior distribution),而是基于抽样分布(sampling distribution)的概念.这种分布中,估计器(estimator)在用于不同的数据集的时候,从真实的未知分布中进行抽样,具体细节参考本书6.2.重复试验的变化的概念就构成了使用频率论方法来对不确定性建模的基础.

相比之下,在贝叶斯方法中,只接受被实际观察的数据,而并没有重复测试的概念.这就允许贝叶斯方法用于计算单次事件的概率,比如在本书2.1中讲的.另一方面可能更重要,就是贝叶斯方法能避免一些困扰了频率论方法的悖论(参考本书6.6).不过总还是要熟悉一下频率论的(尤其是本书的6.5),因为这种方法在机器学习中应用也很广泛的.

## 6.2 一个估计器的抽样分布(Sampling distribution of an estimator)

在频率论统计学中,参数估计$\hat\theta$是通过对某个数据集D来使用一个估计器(estimator)$\delta$而计算得到的,也就是$\hat\theta=\delta(D)$.这里参数被当做固定的,而数据可以是随机的,正好和贝叶斯方法中的完全相反.参数估计的不确定性可以通过计算估计器的抽样分布(sampling distribution)来衡量.要理解这个概念,可以设想从来自某个真实模型($p(*|\theta^*)$)的多个不同的数据集$D^{(s)}$中抽样,设$D^{(s)}= \{x_i^{(s)}\}^N_{i=1}$,其中$x_i^{s}\sim p(*|\theta^*)$,而$\theta^*$是真实参数.而$s=1:S$是对抽样数据集的索引,而N是每一个这样的数据集的规模.然后将估计器$\hat\theta(*)$用于每个$D^{(s)}$来得到一系列的估计$\{\hat\theta(D^{(s)}\}$.然后设$S\rightarrow \infty$,$\hat\theta(*)$就是估计器的抽样分布.接下来的章节中我们会介绍很多种应用这个抽样分布的方法.不过首先还是展示两种方法来计算这个抽样分布本身.


此处参考原书图6.1

### 6.2.1 Bootstrap

Bootstrap是一种简单的蒙特卡罗方法,来对抽样分布进行近似.在估计器是真实参数的复杂函数的情况下特别有用.

这种方法的思想很简单.如果我们知道了真实参数$\theta(*)$,就可以声称很多个,比如S个假的数据结构,每个规模都是N,都是来自于真实分布$x^s_i \sim p(*|\theta^*)$,其中$s=1:S,i=1:N$.然后就可以从每个样本来计算估计器$\hat\sigma^s =f(x^s_{1:N})$,然后使用所得样本的经验分布作为我们对抽样分布的估计。由于$\theta$是未知的,参数化Bootstrap方法的想法是使用$\theta(D)$作为替代来生成样本.另一种方法叫非参数化的Bootstrap方法,是对$x_i^s$从原始数据D中进行可替代抽样,然后按照之前的方法计算诱导分布(induced distribution).有的方法可以在大规模数据集的场景下对Bootstrap进行加速,具体参考 (Kleiner et al. 2011).

图6.1展示了一例,其中使用了参数化Bootstrap来计算一个伯努利分布的最大似然估计(MLE)的抽样分布.(使用非参数化Bootstrap的结果本质上是相同的.)可以看到,当N=10的时候,抽样分布是不对称的,所以很不像高斯分布;而当N=100的时候,这个分布就看上去更像高斯分布了,也正如下文中所述的.

很自然的一个问题是:使用Bootstrap计算出来的参数估计器$\theta^s =\hat\theta(x^s_{1:N})$和采用后验分布抽样得到的参数值$\theta^s\sim p(*|D)$有啥区别呢?
概念上两者很不一样,不过一般情况下,在先验不是很强的时候,这两者可能很相似.例如图6.1(c-d)所示就是一例,其中使用了均匀$\beta$分布$Beta(1,1)$作为先验来计算的后验,然后对其进行抽样.从图中可以看出后验和抽样分布很相似.所以有人可能就认为Bootstrap分布就可以当做是"穷人的"后验;更多内容参考(Hastie et al. 2001, p235).

然而很悲伤,Bootstrap可能会比后验取样要慢很多.原因就是Bootstrap必须对模型拟合S次,而在后验抽样中,通常只要对模型拟合一次(来找到局部众数,local mode),然后就可以在众数周围进行局部探索(local exploration).这种局部探索(local exploration)通常要比从头拟合模型快得多.



### 6.2.2 最大似然估计(MLE)的大样本理论(Large sample theory)

有时候有的估计器(estimator)的抽样分布可以以解析形式计算出来.比如在特定条件下,随着抽样规模趋向于无穷大,最大似然估计(MLE)的抽样分布就成了高斯分布了.简单来说,要得到这个结果,需要模型中每个参数都能"观察"到无穷多个数据量,然后模型还得是可识别的(identifiable).很不幸,这种条件是很多机器学习中常用模型都无法满足的.不过咱们还是可以假设一个简单的环境,使定理成立.

这个高斯分布的中心就是最大似然估计(MLE)$\theta$了.但方差是啥呢?直觉告诉我们这个估计器的方差可能和似然函数面(likelihood surface)的峰值处的曲率(curvature)有关(也可能是负相关).如果曲率很大,峰值那里就很陡峭尖锐,方差就小;这时候就认为这个估计确定性好(well determined).反之如果曲率很小,峰值就几乎是平的,那方差就大了去了.

咱们将这种直观感受用正规数学语言表达一下.定义一个得分函数(score function),也就是对数自然函数在某一点$\theta$处的梯度(gradient):

$s(\hat\theta)\overset{\triangle}{=} \nabla \log p(D|\theta)|_{\hat\theta} $(6.1)

把负得分函数(negative score function)定义成观测信息矩阵(observed information matrix),等价于负对数似然函数(Negative Log Likelihood,缩写为NLL)的海森矩阵(Hessian):

$J(\hat\theta(D))\overset{\triangle}{=} -\nabla s(\hat\theta)=-\nabla^2_\theta \log p(D|\theta)|_{\hat \theta}$(6.2)

在1维情况下,就成了:

$J(\hat\theta(D))=-\frac{d}{d\theta^2}\log p(D|\theta)|_{\hat\theta}$(6.3)

这就是对对数似然函数在点$\hat\theta$位置曲率的一种度量了.

由于我们要研究的是抽样分布,$D=(x_1,...,x_N)$是一系列随机变量的集合.那么费舍信息矩阵(Fisher information matrix)定义就是观测信息矩阵(observed information matrix)的期望值(expected value):

$I_N(\hat\theta|\theta^*) \overset{\triangle}{=}  \mathrm{E}_{\theta^*}[J(\hat\theta|D)]$(6.4)

其中的$ \mathrm{E}_{\theta^*}[f(D)] \overset{\triangle}{=} \frac{1}{N} \sum^N_{i=1}f(x_i)p(x_i|\theta^*)$ 是将函数f用于从$\theta^*$中取样的数据时的期望值.通常这个$\theta^*$表示的都是生成数据的"真实参数",假设为已知的,所以就可以缩写出$I_N(\hat\theta)\overset{\triangle}{=} I_N(\hat\theta|\theta^*)$.另外,还很容易就能看出$I_N(\hat\theta)=NI_1(\hat\theta)$,因为规模为N的样本对数似然函数自然要比规模为1的样本更加"陡峭(steeper)".所以可以去掉那个1的下标(subscript),然后就只写成$I_N(\hat\theta)=I_1(\hat\theta)$.这是常用的表示方法.

然后设最大似然估计(MLE)为$\hat\theta \overset{\triangle}{=}\hat\theta_{mle}(D)$,其中的$D\sim\theta^*$.随着$N \rightarrow \infty$,则有(证明参考Rice 1995, p265)):

$\hat\theta \rightarrow N((\theta^*,I_N(\theta^*)^{-1})$(6.5)

我们就说这个最大似然估计(MLE)的抽样分布是渐进正态(asymptotically normal)的.

那么最大似然估计(MLE)的方差呢?这个方差可以用来衡量对最大似然估计的信心量度.很不幸,由于$\theta^*$是位置的,所以咱们不能对抽样分布的方差进行估计.不过还是可以用$\hat\theta$替代$\theta^*$来估计抽样分布.这样得到的$\hat\theta_k$近似标准差(approximate standard errors)为:

$\hat{se}_k \overset{\triangle}{=} I_N(\hat\theta)_{kk}^{-\frac{1}{2}} $(6.6)

例如,从等式5.60就能知道一个二项分布模型(binomial sampling model)的费舍信(Fisher information)为:

$I(\theta)=\frac{1}{\theta(1-\theta)}$(6.7)

然后最大似然估计(MLE)的近似标准差(approximate standard error)为:

$\hat{se} = \frac{1}{\sqrt{I_N(\hat\theta)}} = \frac{1}{\sqrt{NI(\hat\theta)}}=(\frac{\hat\theta (1-\hat\theta)}{N})^{\frac{1}{2}}$(6.8)

其中$\hat\theta =\frac{1}{N}\sum_iX_i$.可以对比等式3.27,即均匀先验下的后验标准偏差(posterior standard deviation).


## 6.3 频率论决策理论(Frequentist decision theory)

在频率论或者所为经典决策理论中,有损失函数和似然函数,但没有先验,也没有后验,更没有后验期望损失(posterior expected loss)了.因此和贝叶斯方法不同,频率论方法中没有办法来自动推导出一个最优估计器.在频率论方法中,可以自由选择任意的估计器或者决策规则$\delta:X\rightarrow A$.
选好了估计器,就可以定义对应的期望损失(expected loss)或者风险函数(risk),如下所示:
$R(\theta^*,\delta)\overset{\triangle}{=} \mathrm{E} _{p(\tilde D|\theta^*)}[L(\theta^*,\delta(\tilde D))=\int L(\theta^*,\delta(\tilde D))p(\tilde D|\theta^*)d\tilde D]$(6.9)

上式中的$\tilde D$是从"自然分布(nature’s distribution)"抽样的数据,用参数$\theta^*$来表示.也就是说,期望值是估计量大抽样分布相关的.可以和贝叶斯后验期望损失(Bayesian posterior expected loss:)相比:

$\rho(a|D,\pi) \overset{\triangle}{=}  \mathrm{E}[L(\theta,a)]=\int_\Theta L(\theta,a)p(\theta|D,\pi)d\theta   $(6.10)

很明显贝叶斯方法是在位置的$\theta$上进行平均,条件为已知的D,而频率论方法是在$\tilde D$上平均,(也就忽略了观测值),而条件是未知的$\theta^*$.


这种频率论的定义不光看着很不自然,甚至根本就没办法计算,因为$\theta^*$都不知道.结果也就不能以频率论的风险函数(frequentist risk)来对比不同的估计器了.接下来就说一下对这个问题的解决方案.

### 6.3.1 贝叶斯风险

怎么挑选估计器呢?我们需要把$R(\theta^*,\delta)$转换成一个不需要依赖$\theta^*$的单独量$R(\delta)$.一种方法是对$\theta^*$设一个先验,然后定义一个估计器的贝叶斯风险(Bayes risk)或者积分风险(integrated risk),如下所示:

$R_B(\delta) \overset{\triangle}{=}  \mathrm{E}_{p(\theta^*)}[R(\theta^*,\delta)]=\int R(\theta^*,\delta)p(\theta^*)d \theta^* $(6.11)

贝叶斯估计器(Bayes estimator)或者贝叶斯决策规则(Bayes decision rule)就是将期望风险最小化:
$\delta_B \overset{\triangle}{=} \arg\min_\delta R_B(\delta) $(6.12)

要注意这里的积分风险函数(integrated risk)也叫做预制后验风险(preposterior risk),因为实在看到数据之前得到的.对此最小化有助于实验设计.

接下来有一个重要定理,这个定理将贝叶斯方法和频率论方法一起结合到了决策理论中.

#### 定理6.31

贝叶斯估计器可以通过最小化每个x的后验期望损失(posterior expected loss)而得到.

证明.切换积分顺序,就有:
$$
\begin{aligned}
R_B(\delta)& = \int [\sum_x\sum_y L(y,\delta(x))p(x,y|\theta^*)]p(\theta^*)d\theta^* &\text{(6.13)}\\
&\sum_x\sum_y \int_\Theta L(y,\delta(x))p(x,y,\theta^*)d\theta^* &\text{(6.14)}\\
& =\sum_x[\sum_y L(y,\delta(x))p(y|x)dy]p(x) &\text{(6.15)}\\
& =\sum_x \rho (\delta(x)|x)p(x) &\text{(6.16)}\\
\end{aligned}
$$

此处参考原书图6.2

要最小化全局期望(overall expectation),只要将每个x项最小化就可以了,所以我们的决策规则就是要挑选:

$\delta_B (x)=\arg\min_{a\in A} \rho(a|x) $(6.17)
证明完毕.


#### 定理6.32 (Wald,1950)

每个可接受的决策规则都是某种程度上的贝叶斯决策规则,对应着某些可能还不适当的先验分布.
这就表明,对频率论风险函数最小化的最佳方法就是贝叶斯方法!更多信息参考(Bernardo and Smith 1994, p448).

### 6.3.2 最小最大风险(Minimax risk)

当然咯,很多频率论者不喜欢贝叶斯风险,因为这需要选择一个先验(虽然这只是对估计器的评估中要用到,并不影响估计器的构建).所以另外一种方法就如下所示.定义一个估计器的最大风险如下所示:

$R_{max}(\delta) \overset{\triangle}{=} \max_{\theta^*} R(\theta^*,\delta)$(6.18)

最小最大规则(minimax rule)就是将最大风险最小化:
$\delta_{MM}\overset{\triangle}{=} \arg\min_\delta R_{max}(\delta)$(6.19)

例如图6.2中,很明显在所有的$\theta^*$值上,$\delta_1$有比$\delta_2$更低的最差情况风险(lower worst-case risk),所以就是最小最大估计器(关于如何计算一个具体模型的风险函数的解释可以参考本书6.3.3.1).

最小最大估计器有一定的吸引力.可惜,计算过程可难咯.而且这些函数还都很悲观(pessimistic).实际上,所有的最小最大估计器都是等价于在最不利先验下的贝叶斯估计器.在大多数统计情境中(不包括博弈论情境),假设自然充满敌意并不是一个很合理的假设.

