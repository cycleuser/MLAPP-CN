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

那么最大似然估计(MLE)的方差呢?这个方差可以用来衡量对最大似然估计的信心量度.很不幸,由于$\theta^*$是未知的,所以咱们不能对抽样分布的方差进行估计.不过还是可以用$\hat\theta$替代$\theta^*$来估计抽样分布.这样得到的$\hat\theta_k$近似标准差(approximate standard errors)为:

$\hat{se}_k \overset{\triangle}{=} I_N(\hat\theta)_{kk}^{-\frac{1}{2}} $(6.6)

例如,从等式5.60就能知道一个二项分布模型(binomial sampling model)的费舍信息(Fisher information)为:

$I(\theta)=\frac{1}{\theta(1-\theta)}$(6.7)

然后最大似然估计(MLE)的近似标准差(approximate standard error)为:

$\hat{se} = \frac{1}{\sqrt{I_N(\hat\theta)}} = \frac{1}{\sqrt{NI(\hat\theta)}}=(\frac{\hat\theta (1-\hat\theta)}{N})^{\frac{1}{2}}$(6.8)

其中$\hat\theta =\frac{1}{N}\sum_iX_i$.可以对比等式3.27,即均匀先验下的后验标准偏差(posterior standard deviation).


## 6.3 频率论决策理论(Frequentist decision theory)

在频率论或者经典决策理论中,有损失函数和似然函数,但没有先验,也没有后验,更没有后验期望损失(posterior expected loss)了.因此和贝叶斯方法不同,频率论方法中没有办法来自动推导出一个最优估计器.在频率论方法中,可以自由选择任意的估计器或者决策规则$\delta:X\rightarrow A$.
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

要注意这里的积分风险函数(integrated risk)也叫做预制后验风险(preposterior risk),因为是在看到数据之前得到的.对此最小化有助于实验设计.

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




### 6.3.3 可容许的估计器(Admissible estimators)

频率论决策理论的基本问题就是要知道真实分布$p(*|\theta^*)$才能去评估风险.可是有的估计器可能不管$\theta^*$是什么值,都会比其他的一些估计器更差.比如说,如果对于所有的$\theta\in\Theta$,都有$R(\theta,\delta_1)<R(\theta,\delta_2)$,那么就说$\delta_1$支配了$\delta_2$.如果不等关系对于某个$\theta$来说严格成立,就说这种支配关系是严格的.如果一个估计器不被另外的估计器所支配,就说这个估计器是可容许的(Admissible).

#### 6.3.3.1 样例

这个例子基于(Bernardo and smith 1994).这个问题是去估计一个正态分布的均值.假设数据样本抽样自一个正态分布$x_i \sim N(\theta^*,\sigma^2=1)$,然后使用平方损失函数(quadratic loss)$L(\theta,\hat\theta)=(\theta-\hat \theta)^2$.这时候对应的风险函数就是均方误差(MSE).估计器$\hat\theta(x)=\delta(x)$也就是一些可能的决策规则,如下所示:

* $\delta_1(x)=\bar x$,这个是样本均值
* $\delta_2(x)=\tilde x$,这个是样本中位数
* $\delta_3(x)=\theta_0$,这个是某个固定值
* $\delta_k(x)$,这个是在先验$N(\theta|\theta_0,\sigma^2/k)$下的后验均值:
$\delta_k(x)=\frac{N}{N+k}\bar x  + \frac{k}{N+k}\theta_0 =w\bar x +(1-w)\theta_0$(6.20)

对$\delta_k$可以设置一个弱先验$k=1$,以及一个更强的先验$k=5$.先验均值是某个固定值$\theta_0$.然后假设$\sigma^2$已知.(这样$\delta_3(x)$就和$\delta_k(x)一样了,后者有一个无穷强先验$k=\infty$.)

接下来就以解析形式来推导风险函数了.(在这个样例中可以这么做,是因为已经知道了真实参数$\theta^*$.)在本书6.4.4,会看到均方误差(MES)可以拆解成平方偏差(squared bias)加上方差(variance)的形式:
$MSE(\hat\theta(*)|ptheta^*$ = var[\hat\theta]+bias^2(\hat\theta)(6.21)

样本均值是没有偏差的(unbiased),所以其风险函数为:
$MSE(\delta_1|\theta^*) =var[\bar x]=\frac{\sigma^2}{N} $(6.22)

此处参考原书图6.3


样本中位数也是无偏差的.很明显其方差大约就是$\pi/(2N)$,所以有:

$MSE(\delta_2|\theta^*)=\frac{\pi}{2N} $(6.23)


对于固定值的$\delta_3(x)=\theta_0$,方差是0,所以有:

$MSE(\delta_3|\theta^*)= (\theta^*-\theta_0)^2  $(6.24)

最后是后验均值,如下所示:

$$
\begin{aligned}
MSE(\delta_k|\theta^*)&= \mathrm{E}[ (w\bar x+(1-w)\theta_0-\theta^*)^2      ]   &\text{(6.25)}\\
&=\mathrm{E}[(w(\barx-\theta^*)+(1-w)(\theta_0-\theta^*)        )^2]    &\text{(6.26)}\\
&=  w^2\frac{\sigma^2}{N}+(1-w)^2(\theta_0-\theta^*)^2  &\text{(6.27)}\\
&=  \frac{1}{(N+k)^2}(N\sigma^2+k^2(\theta_0-\theta^*)^2)  &\text{(6.28)}\\
\end{aligned}
$$

图6.3中所示是在$N\in \{5,20\}$范围内的上面几个函数的图像.可以看出总体上最佳的估计器都是依赖$\theta^*$值的那几个,可是$\theta^*$还是未知的.如果$\theta^*$很接近$\theta_0$,那么$\delta_3$(实际上就是预测的$\theta_0$)就是最好的了.如果$\theta^*$在$\theta_0$范围内有一定波动,那么后验均值,结合了对$\theta_0$的猜测和数据所反映的信息,就是最好的.如果$\theta^*$远离$\theta_0$,那么最大似然估计(MLE)就是最好的.这也一点都不让人意外:假设先验均值敏感,小规模的收缩(shrinkage)是通常期望的(使用一个弱先验的后验均值).

令人意外的是对于任意的$\theta_0$值,决策规则$\delta_2$(样本中位数)的风险函数总是比$\delta_1$(样本均值)的风险函数更大.也就是说,样本中位数对于这个问题来说是一个不可容许估计器(inadmissible estimator)(这个问题是假设数据抽样自一个正态分布).

可是在实践中,样本中位数其实往往比样本均值更好,因为对于异常值不敏感,更健壮.参考(Minka 2000d)可知,如果假设数据来自于比高斯分布(正态分布)更重尾(heavier tails)的拉普拉斯分布(Laplace distribution),那么样本中位数就是贝叶斯估计器(Bayes estimator)(使用平方损失函数(squared loss)).更一般地,可以对数据使用各种灵活的模型来构建健壮的估计器,比如混合模型,或者本书14.7.2会讲到的非参数化密度估计器(non-parametric density estimators),然后再去计算后验均值或者中位数.

#### 6.3.3.2 斯坦因悖论(Stein’s paradox)*


加入有N个独立同分布(iid)随机变量服从正态分布,即$X_i\sim N(\theta_i,1)$,然后想要估计$\theta_i$.很明显估计器应该用最大似然估计(MLE)这时候就是设$\hat\theta_i=x_i$.结果表明:$N\ge 4$的时候,这是一个不可容许估计器(inadmissible estimator).

怎么证明?构建一个更好的估计器就可以了.比如吉姆斯-斯坦因估计器(James-Stein estimator)就可以,定义如下所示:

$\hat\theta_i =\hat B \bar x +(1-\hat B)(x_i-\bar x)$(6.29)

上式中的$\bar x=\frac{1}{N}\sum^N_{i=1}x_i$,而$0<B<1$是某个调节常数.这个估计其会将$\theta_i$朝向全局均值进行收缩(shrink).(在本书5.6.2使用经验贝叶斯方法推导过这个估计器.)

很明显,当$N\ge 4$的时候,这个收缩估计器比最大似然估计(MLE,也就是样本均值)有更低的频率风险(均方误差MSE).这就叫做斯坦因悖论(Stein's paradox).为啥说这是个悖论呢?这就解释一下.假如$\theta_i$是某个学生i的真实智商值(IQ),而$X_i$是其测试分数.为啥用全局均值来估计特定的$\theta_i$,用其他学生的分数去估计另一个学生的分数?利用不同范畴的东西还可以制造更加荒诞的自相矛盾的例子,比如加入$\theta_1$是我的智商,而$\theta_2$是温哥华平均降雨量,这有毛线关系?

这个悖论的解决方案如下所述.如果你的目标只是估计$\theta_i$,那么用$x_i$来估计就已经是最好的选择了,可是如果你的目的是估计整个向量$\theta$,而且你还要用平方误差(squared error)作为你的损失函数,那么那个收缩估计器就更好.为了更好理解,假设我们要从一个单一样本$x\sim N(\theta,I)$来估计$||\theta||^2_2$.最简单的估计就是$||x||^2_2$,不过会遇到上面的问题,因为:


$\mathrm{E}[||x||^2_2]=\mathrm{E}[\sum_i x^2_i] =\sum^N_{i=1}(1+\theta_i^2)=N+||\theta||^2_2     $(6.30)

结果就需要增加更多信息来降低风险,而这些增加的信息可能甚至来自于一些不相关的信息源,然后估计就会收缩到全局均值上面了.在本书5.6.2对此给出过贝叶斯理论的解释,更多细节也可以参考(Efron and Morris 1975).


#### 6.3.3.3 可容许性远远不够(Admissibility is not enough)

从前文来看,似乎很明显我们应该在可容许估计器范围内来搜索好的估计器.但实际上远不止构建可容许估计器这么简单,比如接下来这个例子就能看出.

#### 定理 6.3.3

设有正态分布$X\sim N(\theta,1)$,在平方误差下对$\theta$进行估计.设$\delta_1(x)=\theta_0$是一个独立于数据的常量.这是一个可容许估计器(admissible estimator).

证明:用反证法,假设结论不成立,存在另外一个估计器$\delta_2$有更小风险,所以有$R(\theta^*,\delta_2)\le R(\theta^*,\delta_1)$,对于某些$\theta^*$不等关系严格成立.设真实参数为$\theta^*=\theta_0$.则$R(\theta^*,\delta_1)=0$,并且有:

$R(\theta^*,\delta_2)=\int (\delta_2(x)-\theta_0)^2p(x|\theta_0)dx$(6.31)

由于对于所有的$\theta^*$都有$0\le R(\theta^*,\delta_2)\le R(\theta^*,\delta_1)$,而$R(\theat_0,\delta_1)=0$,所以则有$R(\theta_0,\delta_2)=0,\delta_2(x)=\theta_0=\delta_1(x)$.这就表明了$\delta_2$只有和$\delta_1$相等的情况下才能避免在某一点$\theta_0$处有更高风险.也就是说不能有其他的估计器$\delta_2$能严格提供更低的风险.所以$\delta_1$是可容许的.证明完毕



## 6.4 估计器的理想性质

由于频率论方法不能提供一种自动选择最佳估计器的方法,我们就得自己想出其他启发式的办法来进行选择.在本节,就讲一下我们希望估计器所具有的一些性质.不过很不幸,我们会发现这些性质不能够同时满足.

### 6.4.1 连续估计器(Consistent estimators)


连续估计器,就是随着取样规模趋近于无穷大,最终能够恢复出生成数据的真实参数的估计器,也就是随着$|D|\rightarrow \infity$,$\hat\hteta(D)\rightarrow \theta^*$(这里的箭头指的是概率收敛的意思).当然了,这个概念要有意义,就需要保证数据确实是来自某个具有参数$\theta^*$的特定模型,而现实中这种情况很少见的.不过从理论上来说这还是个很有用的性质.

最大似然估计(MLE)就是一个连续估计器.直观理解就是因为将似然函数最大化其实就等价于将散度$KL(p(*|\theta^*)||p(*|\hat\theta))$最小化,其中的$p(*|\theta^*)$是真实分布,而$p(*|\hat\theta)$是估计的.很明显当且仅当$\hat\theta=\theta^*$的时候才有0散度(KL divergence).


### 6.4.2 无偏差估计器(Unbiased estimators)

估计器的偏差(bias)定义如下:

$bias(\hat\theta(*)) =\mathrm{E}_{p(D|\theta_*)} [\hat\theta(D)-\theta_* ]   $(6.32)

上式中的$\theta_*$就是真实的参数值.如果偏差为0,就说这个估计器无偏差.这意味着取样分布的中心正好就是真实参数.例如对于高斯分布均值的最大似然估计(MLE)就是无偏差的:

$bias(\hat\mu)  =\mathrm{E}[\bar x]-\mu= =\mathrm{E}[\frac{1}{N{\sum^N_{i=1}x_i] -\mu =\frac{N\mu}{N}-\mu=0  $(6.33)

不过对高斯分布方差$\hat\sigma^2$的最大似然估计(MLE)就不是对$\sigma^2$的无偏估计.实际上可以发现(参考练习6.3):

$\mathrm{E} [\hat\sigma^2]=\frac{N-1}{N}\sigma^2$(6.34)


不过下面这个估计器就是一个无偏差估计器:

$\hat\sigma^2_{N-1}=\frac{N}{N-1}\hat\sigma^2=\frac{1}{N-1}\sum^N_{i=1}(x_i-\bar x)^2$(6.35)

对上式,可以证明有:

$\mathrm{E} [\hat\sigma^2_{N-1}]=\mathrm{E} [\frac{N}{N-1}\sigma^2]=\frac{N}{N-1}\frac{N-1}{N}\sigma^2=\sigma^2  $(6.36)

在MATLAB中,$\var(X)$返回的就是$\hat\simga^2_{N-1}$,而$\var(X,1)$返回的是最大似然估计(MLE)$\sigma^2$.对于足够大规模的N,这点区别就可以忽略了.

虽然最大似然估计(MLE)可能有时候有偏差,不过总会逐渐无偏差.(这也是最大似然估计(MLE)是连续估计器的必要条件.)

虽然无偏听上去好像是个很理想的性质,但也不总是好事,更多细节可以参考本书6.4.4以及(Lindley 1972)的相关讨论.

### 6.4.3 最小方差估计器


直观来看,好像让估计器尽量无偏差是很合理的(不过后面我们会看到事实并非如此简单).不过只是无偏还不够用.比如我们想要从几何$D=\{x_1,..,x_N\}$估计一个高斯均值.最开始对第一个数据点用这个估计器的时候$\hat\theta(D)=x_1$,这时候还是无偏估计,但逐渐就会比经验均值$\bar x$(这也是无偏的)更远离真实的$\theta_*$.所以一个估计器的方差也很重要.

很自然的问题:方差能到多大?有一个著名的结论,叫做克莱默-饶下届(Cramer-Rao lower bound)为任意的无偏估计器的方差提供了下届.具体来说如下所示:

#### 定理6.4.1 (克莱默-饶 不等式)

设 $X_1,..,X_n \sim p(X|\theta_0)$,而$\hat\theta=\hat\theta(x_1,..,x_n)$是一个对参数$\theta_0$的无偏估计器.然后在对$p(X|\theta_0)$的各种平滑假设下,有:

$\var [\hat\theta]\ge \frac{1}{nI(\theta_0)}$(6.37)

其中的$I(\theta_0)$是费舍信息矩阵(Fisher information matrix)(参考本书6.2.2).

对此的证明可以参考(Rice 1995, p275).
可以证明最大似然估计(MLE)能达到克莱默-饶下届,因此对于任意无偏估计器都会有渐进的最小方差.所以说最大似然估计(MLE)是渐进最优(asymptotically optimal)的.


### 6.4.4 偏差-方差权衡 

使用无偏估计看上去是个好主意,实际并非如此简单.比如用一个二次损失函数为例.如上文所述,对应的风险函数是均方误差(MSE).然后我们能推出一个对均方误差(MSE)的有用分解.(所有期望和方差都是关于真实分布$p(D|\theta^*)$,但为了表达简洁,这里就把多余的条件都舍掉了.)设$\hat\theta =\hat \theta(D)$表示这个估计,然后$\bar \theta =\mathrm{E}[\hat\theta]表示的是估计的期望值(变化Ｄ来进行估计)．然后就有：

$$
\begin{aligned}
\mathrm{E} [(\hat\theta-\theta^*)^2 ]&=\mathrm{E} [[(\hat\theta-\bar \theta)+(\bar\theta-\theta^*) ]^2]  &\text{(6.38)\\
&=\mathrm{E} [(\hat\theta-\bar \theta)^2] +2(\bar\theta-\theta^*)\mathrm{E}[\hat\theta-\theta^*]+(\bar\theta-\theta^*)^2  &\text{(6.39)\\
&=\mathrm{E}[(\hat\theta-\bar\theta)^2]+(\bar\theta-\theta^*)^2   &\text{(6.40)\\
&=\var[\hat\theta]+bias^2(\hat\theta)   &\text{(6.41)\\
\end{aligned}
$$

用文字表达就是：

$MSE= variance +bias^2$(6.42)重要公式

这也就是偏差－方差之间的权衡（bias-variance tradeoff），可以参考(Geman et al. 1992)．这就意味着架设我们的目标是要最小化平方误差，那么选择一个有偏差估计器也可能是可取的，只要能够降低方差．

##### 6.4.4.1 样例:估计高斯均值

然后举个例子吧，基于(Hoff 2009, p79).假如要从$x=(x_1,..,x_N)$估计高斯均值.架设数据采样自一个正态分布$x_i\sim N(\theta^*=1,\sigma^2)$.很明显可以用最大似然估计(MLE).这样估计器偏差为0,方差为:

$\var[\bar x|\theta^*] =\frac{\sigma^2}{N}  $(6.43)


不过也可以使用最大后验估计(MAP estimate).在本书4.6.1中,我们已经遇到过使用正态分布$N(\theta_0,\sigma^2/K_0)$先验的最大后验估计为:


$\tilde x \overset{\triangle}{=} \frac{N}{N+k_0}\bar x+\frac{k_0}{N+k_0}\theta_0=w\bar x+(1-w)\theta_0  $(6.44)

其中$0\le w \le 1$控制了我们对最大似然估计(MLE)相比先验的信任程度.(这也是后验均值,因为高斯分布的均值和众数相等.)偏差和方差为:

$$
\begin{aligned}
\mathrm{E}[\tilde x]-\theta^* &=  w\theta_0+(1-w)\theta_0-\theta^*=(1-w)(\theta_0-\theta^*)   &\text{(6.45)\\
\var[\tilde x]&= w^2\frac{\sigma^2}{M}  &\text{(6.46)\\
\end{aligned}
$$


此处参考原书图6.4


虽然最大后验估计有偏差(设 w<1),但方差更低.

假设先验有些错误,所以使用$\theta_0=0$,而真实的$\theta^*=1$.如图6.4(a)所示,可以看到对$k_0>0$的最大后验估计的抽样分布是偏离于真实值的,但方差比最大似然估计(MLE)的更低(也就是更窄).

如图6.4(b)所示是$mse(\tilde x)/mse(\bar x)$对 N 的函数曲线.可见最大后验估计(MAP)比最大似然估计(MLE)有更低的均方误差(MSE),尤其是样本规模小的时候,比如$k_0\in \{1,2\}$.$k_0=0$的情况对应的就是最大似然估计(MLE),而$k_0=3$对应的就是强先验,这就会影响性能了,因为先验均值是错的.很明显,对先验强度的调整很重要,后面会讲到.

#### 6.4.4.2 样例:岭回归(ridge regression)


在偏差方差之间进行权衡的另外一个重要例子就是岭回归(ridge regression),我们会在本书7.5讲到.简单来说,对应的就是在高斯先验$p(w)=N(w|0,\lambda^{-1}I$下对线性回归(linear regression)的最大后验估计(MAP).零均值先验使得先验的权值很小,也就降低了过拟合的概率;精度项$\lambda$控制了先验的强度.设置$\lambda=0$就是最大似然估计(MLE);使用$\lambda>0$就得到了一个有偏差估计.要展示方差的效果,可以考虑一个简单的例子.比如图6.5左边的就是每个拟合曲线的投图,右边的是平均的拟合曲线.很明显随着归一化强度的增加,方差降低了,但是偏差增大了.


此处参考原书图6.5

#### 6.4.4.3 对于分类的偏差-方差权衡

如果使用0-1损失函数,而不是用平方损失,上面的分析就不适合了,因为频率论的风险函数不再能表达成平方偏差加上方差的形式了.实际上可以发现(参考练习7.2(Hastie et al. 2009)):偏差和方差是以相乘方式结合起来的(combine multiplicatively).如果估计结果处于决策边界的正确一侧,偏差就是负的,然后降低方差就可以降低误分类率.但如果估计位于决策边界的错误一侧,那么偏差就是正的,就得增加方差(Friedman 1997a).这就表明对于分类来说,偏差方差权衡没多大用.最好还是关注到期望损失(expected loss)上,而不是直接关注偏差和方差.可以使用交叉验证来估计期望损失,具体会在本书6.5.3中讲到.




