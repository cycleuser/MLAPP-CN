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

#### 9.2.2.4 反面样本(Non-examples)

肯定不可能所有分布都属于指数族啊.比如均匀分布$X\sim Unif(a,b)$就不属于指数族,因为这个分布的支撑(support,其实大概也就是定义域的意思)是一来参数的.另外本书11.4.5所示的学生T分布也不属于指数族,因为不含有要求的形式.


### 9.2.3 对数分区函数(log partition function)

指数族的一个重要性质就是对数分区函数的导数(derivatives)可以用来生成充分统计的累积量(cumulants).由于这个原因才使得$A(\theta)$有时候也被叫做累积函数(cumulant function).先对一个单参数分布来证明一下;然后可以直接泛化扩展到K个参数的分布上.首先是求一阶导数得到了:
$$
\begin{aligned}
\frac{dA}{d\theta}&   =  \frac{d}{d\theta} (\log \int \exp (\theta \phi(x)) h(x)dx)   &\text{(9.27)}\\
& = \frac{\frac{d}{d\theta} \int \exp(\theta\phi(x))h(x)dx}{\int \exp(\theta\phi(x))h(x)dx}                          &\text{(9.28)}\\
& =  \frac{\int \phi(x) \exp(\theta\phi(x))h(x)dx } {\exp(A(\theta)}                        &\text{(9.29)}\\
& = \int \phi(x)\exp(\theta\phi(x)-A(\theta)) h(x)d x                 &\text{(9.30)}\\
& =  \int \phi(x)p(x)dx =\mathrm{E}[\phi(x)]       &\text{(9.31)}\\
\end{aligned}
$$

然后求二阶导数就得到了:

$$
\begin{aligned}
\frac{d^2A}{d\theta ^2} &=  \int \phi(x)\exp(\theta\phi(x)-A(\theta))h(x)(\phi(x)-A'(\theta))dx &\text{(9.32)}\\
&= \int \phi(x)p(x)(\phi(x)-A'(\theta))dx &\text{(9.33)}\\
&= \int \phi^2(x)p(x)dx-A'(\theta)\int \phi(x)p(x)dx &\text{(9.34)}\\
&=  \mathrm{E}[\phi^2(X)] -\mathrm{E}[\phi(x)]^2 =var[\phi(x)] &\text{(9.35)}\\
\end{aligned}
$$

其中利用了$A'(\theta)=\frac{dA}{d\theta} =\mathrm{E}[\phi(x)]$.

在多变量的情况下,有:

$\frac{\partial ^2 A}{\partial\theta_i \partial\theta_j}  = \mathrm{E}[\phi_i(x)\phi_j)(x)] - \mathrm{E}[\phi_i(x)] \mathrm{E}[\phi_j(x)]  $(9.36)

然后就有:

$\nabla ^2 A(\theta) =cov [\phi(x)]$(9.37)

因此协方差矩阵就是正定的,会发现$A(\theta)$是一个凸函数(参考本书7.3.3).

#### 9.2.3.1 样例:伯努利分布

以伯努利分布为例.$A(\theta) =\log(1+e^\theta)$,这样就有均值为:
$\frac{dA}{d\theta} =\frac{e^\theta}{1+e^\theta} =\frac{1}{1+e^{-\theta}} = sigm(\theta)=\mu $(9.38)

方差为:

$$
\begin{aligned}
\frac{d^2A}{d\theta^2} &=  \frac{d}{d\theta} (1+e^{-\theta})^{-1} =(1+e^{-\theta})^{-2}.e^{-\theta} &\text{(9.39)}\\
&= \frac{e^{-\theta}}{1+e^{-\theta}}\frac{1}{1+e^{-\theta}}=\frac{1}{e^\theta+1}\frac{1}{1+e^{-\theta}}=(1-\mu)\mu &\text{(9.40)}\\
\end{aligned}
$$

### 9.2.4 指数族的最大似然估计(MLE)


指数族模型的似然函数形式如下:

$p(D|\theta)=[\prod^N_{i=1}h(x_i)] g(\theta)^N\exp(\eta(\theta)^T[\sum^N_{i=1}\phi(x_i)])$(9.41)
然后会发现充分统计量(sufficient statistics)为N以及:

$\phi(D)=[\sum^N_{i=1}\phi_1(x_i),...,\sum^N_{i=1}\phi_k(x_i)]$(9.42)

对伯努利模型为$\phi=[\sum_i I(x_i=1)]$,单变量高斯模型则有$\phi =[\sum_i x_i,\sum_i x_i^2]$.(还需要知道样本规模N.)

Pitman-Koopman-Darmois定理表面,在特定规范化条件(regularity conditions)下,指数族分布是唯一有有限充分统计量的分布.(这里的有限(finite)是与数据集规模无关的.)

这个定理需要的一个条件是分布的支撑(support,就当做定义域理解了)不能独立于参数.比如下面这个均匀分布为例:

$p(x|\theta)=U(x|\theta)=\frac{1}{\theta}I(0\le x\le \theta)$(9.43)

似然函数就是:

$p(D|\theta)=\theta^{-N}I(0\le \max\{ x_i\}\le \theta)$(9.44)

所以充分统计量就是N和$s(D)=\max_ix_i$.这个均匀分布规模有限,但并不是指数族分布,因为其支撑集合(support set)X依赖参数.

接下来就要讲下如何对一个规范指数族分布模型(canonical exponential family model)计算最大似然估计(MLE).给定了N个独立同分布的数据点,对数似然函数(log-likelihood）为:

$\log p(D|\theta) =\theta^T\phi(D)-NA(\theta)$(9.45)


由于$-A(\theta)$在$\theta$上是凹的,而$\theta^T\phi(D)$在$\theta$上是线性的,而对数似然函数是凹的,所以这就会有唯一的一个全局最大值.要推导这个最大值,要用到对数分区函数(log partition function)的导数生成充分统计量向量的期望值(参考本书9.2.3):

$\nabla_\theta\log p(D|\theta)=\phi(D)-N\mathrm{E}[\phi(X)]$(9.46)

设置梯度为零,就可以得到最大似然估计(MLE)了,充分统计量的经验均值必须等于模型的理论期望充分统计量,也就是$\hat \theta$必须满足下面的条件:

$\mathrm{E}[\phi(X)]\frac{1}{N}\sum^N_{i=1}\phi(x_i)$(9.47)

这就叫做矩捕获(moment matching).比如在伯努利分布中,就有$\phi(X)=I(X=1)$,所以最大似然估计(MLE)就满足:

$\mathrm{E}[\phi(X)]=p(X=1)=\hat \mu =\frac{1}{N}\sum^N_{i=1}I(x_i=1)$(9.48)

### 9.2.5 指数族的贝叶斯方法*

我们已经看见了,如果先验和似然函数是共轭的,那么确定的贝叶斯分析就相当简单了.粗略地理解也可以认为这意味着先验$p(\theta|\tau)$和似然函数$p(D|\theta)$形式一样.为了好理解,就需要似然函数you有限的充分统计量,所以就可以写成$p(D|\theta)=p(s(D)|\theta)$.这表明只有指数族分布才有共轭先验.接下来推导先验和后验的形式.


#### 9.2.5.1 似然函数

指数族似然函数为:

$p(D|\theta)\propto g(\theta)^N\exp(\eta(\theta)^Ts_N)$(9.49)

其中$s_N =\sum^N_{i=1}s(x_i)$.以规范参数的形式就成了:

$p(D|\eta)\propto \exp(N\eta^T \bar s -NA(\eta))$(9.50)

其中$\bar s=\frac{1}{N} s_N$.

#### 9.2.5.2 先验

自然共轭先验形式如下:

$p(\theta|v_0,\tau_0)\propto g(\theta)^{v_0}\exp(\eta(\theta)^T\tau_0$(9.51)

然后写成$\tau_0=v_0\bar \tau_0$来区分先验中伪数据的规模$v_0$和在这个伪数据上充分统计的均值$\bar \tau_0$.写成规范形式(canonical form),先验就成了:

$p(\eta|v_0,\bar\tau_0)\propto \exp(v_0 \eta^T \bar \tau_0-v_0 A(\eta))$(9.52)

#### 9.2.5.3 后验

后验形式为:
$p(\theta|D)=p(\theta|v_N,\tau_N)=p(\theta|v_0+N,\tau_0+s_N)$(9.53)

可见就可以通过假发来更新超参数(hyper-parameters).用规范性就是:

$$
\begin{aligned}
p(\theta|D)&\propto \exp(\eta^T(v_0\bar \tau_0+N\bar s)-(v_0+N) A(\eta))            &\text{(9.54)}\\
&=  p(\eta|v_0+N,\frac{v_0\bar \tau _0+N \bar s}{v_0+N})  &\text{(9.55)}\\
\end{aligned}
$$

所以就会发现后验超参数是先验均值超参数和充分统计量均值的凸组合(convex combination).

#### 9.2.5.4 后验预测密度

给定已有数据$D=(x_1,...,x_N)$,对未来观测量$D'=(\tilde x_1,...,\tilde x_N)$的预测密度的通用表达式进行推测,过程如下所述.为了记号简单,将充分统计量和数据规模结合起来,即:$\tilde \tau_0 =(v_0,\tau_0),\tilde s(D)=(N,s(D)),\tilde s(D')=(N',s(D'))$.先验就是:

$p(\theta|\tilde \tau_0) =\frac{1}{Z(\tilde \tau_0)} g(\theta)^{v_0} \exp(\eta(\theta)^T\tau_0) $(9.56)

似然函数和后验形式相似.因此有:

$$
\begin{aligned}
p(D'|D)&= \int p(D'|\theta)p(\theta|D)d\theta &\text{(9.57)}\\
&= [\prod^{N'}_{i=1} h(\tilde x_i)]Z(\tilde\tau_0+\tilde s(D))^{-1}\int g(\theta)^{v_0+N+n'}d\theta    &\text{(9.58)}\\
&\times \exp( \sum_k\eta_k(\theta) (\tau_k +\sum^N_{i=1} s_k(x_i)+\sum^{N'}_{i=1}s_k(\tilde x_i) ) )d\theta &\text{(9.59)}\\
&= [\prod^{N'}_{i=1} h(\tilde x_i)]\frac{Z(\tilde \tau_0+\tilde s(D)+ \tilde s(D'))}{Z(\tilde \tau_0+\tilde s(D))}      &\text{(9.60)}\\
\end{aligned}
$$

如果$N=0$,这就成了$D'$的边缘似然函数,降低(reduce)到了通过先验的归一化项乘以一个常数而得到的后验归一化项(normalizer)的类似形式.

#### 9.2.5.5 样例:伯努利分布

举个简单例子,这回用新形式回顾一下$\beta$伯努利分布(Beta-Bernoulli model).

似然函数为:

$p(D|\theta)=(1-\theta)^N\exp(\log(\frac{\theta}{1-\theta})\sum_i x_i)$(9.61)

共轭先验为:

$$
\begin{aligned}
p(\theta|v_0,\tau_0) &\propto (1-\theta)^{v_0}\exp(\log(\frac{\theta}{1-\theta})\tau_0)            &\text{(9.62)}\\
&= \theta^{\tau_0}(1-\theta)^{v_0-\tau_0}     &\text{(9.63)}\\
\end{aligned}
$$

如果定义$\alpha =\tau_0+1,\beta=v_0-\tau_0+1$,就会发现这是一个$\beta$分布.

然后可以推导后验了,如下面所示,其中$s=\sum_iI(x_i=1)$是充分统计量:

$$
\begin{aligned}
p(\theta|D)&\propto  \theta^{\tau_0+s}(1-\theta)^{v_0-\tau_0+n-s}     &\text{(9.64)}\\
&= \theta^{\tau_n}(1-\theta)^{v_n-\tau_n}    &\text{(9.65)}\\
\end{aligned}
$$

后验预测分布的推测过程如下所示.设$p(\theta)=Beta(\theta|\alpha,\beta)$,$s=s(D)$是过去数据(past data)的人头数.就可以预测里一系列未来人头朝上$D'=(\tilde x_1,...,\tilde x_m)$的概率,充分统计量为$s'=\sum^m_{i=1}I(\tilde  x_i=1)$,则后验预测分布为:

$$
\begin{aligned}
p(D'|D)=\int^1_0 p(D'|\theta|Beta(\theta|\alpha_n,\beta_n)d\theta&=            &\text{(9.66)}\\
&=    \frac{\Gamma(\alpha_n+\beta_n)}{\Gamma(\alpha_n)\Gamma(\beta_n)} \int^1_0 \theta^{\alpha_n+t'-1}(1-\theta)^{\beta_n+m-t'-1} d\theta        &\text{(9.67)}\\
&=   \frac{\Gamma(\alpha_n+\beta_n) \Gamma (\alpha_{n+m}) \Gamma (\beta_{n+m})}{\Gamma(\alpha_n)\Gamma(\beta_n) \Gamma (\alpha_{n+m}+\beta_{n+m})}          &\text{(9.68)}\\
\end{aligned}
$$

其中

$$
\begin{aligned}
\alpha_{n+m} &= \alpha_n+s' = \alpha_n+s+s'      &\text{(9.69)}\\
\beta_{n+m} &= \beta_n +(m-s')=\beta+(n-s)+(m-s')    &\text{(9.70)}\\
\end{aligned}
$$


