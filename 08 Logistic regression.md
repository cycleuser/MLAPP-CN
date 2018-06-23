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

### 8.3.2 梯度下降(gradient descent)

无约束优化问题的最简单算法,可能就是梯度下降(gradient descent)了,也叫做最陡下降(Steepest descent).写作下面的形式:

$\theta_{k+1}=\theta_k -\eta_kg_k$(8.8)

其中的$\eta_k$是步长规模(step size)或者也叫学习率(learning rate).梯度下降法的主要问题就是:如何设置步长.这个问题还挺麻烦的.如果使用一个固定的学习率,但又太小了,那收敛就会很慢,但如果要弄太大了呢,又可能最终不能收敛.这个过程如图8.2所示,其中对下面的(凸函数)进行了投图:

$f(\theta)=0.5(\theta_1^2-\theta_2)+0.5(\theta_1)^2$(8.9)

任意设置从(0,0)开始.在图8.2(a)中,使用了固定步长为$\eta =0.1 $;可见沿着低谷部位移动很慢.在图8.2(b)中,使用的是固定步长$\eta =0.6 $;很明显这样一来算法很快就跑偏了,根本就不能收敛了.


此处参考原书图8.3

所以就得像个靠谱的办法来选择步长,这样才能保证无论起点在哪里最终都能收敛到局部最优值.(这个性质叫做全局收敛性(global convergence),可千万别跟收敛到全局最优值弄混淆哈.)通过泰勒定理(Taylor's theorem),就得到了:

$f(\theta+\eta d)\approx f(\theta)+\eta g^Td$(8.10)

其中的d是下降方向.所以如果$\eta$足够小了,则$f(\theta+\eta d)< f(\theta)$,因为梯度会是负值的.不过我们并不希望步长太小,否则就要运算很久才能到达最小值了.所以要选一个能够最小化下面这个项的步长$\eta$:

$\phi(\eta)=f(\theta_k +\eta d_k)$(8.11)

这就叫线性最小化(line minimization)或者线性搜索(line search).有很多种方法来借这个一维优化问题,具体细节可以参考(Nocedal and Wright 2006).

图8.3(a)展示了上面那个简单问题中的线性搜索.不过我们会发现线性搜索得到的梯度下降路径会有一种扭折行为(zig-zag behavior).可以看到其中一次特定线性搜索要满足$\eta_k =\arg \min_{\eta>0}\phi (\eta)$.优化的一个必要条件就是导数为零,即$\phi'(\eta)=0$.通过链式规则(chain rule),$\phi'(\eta)=d^Tg$,其中$g=f'(\theta+\eta d)$是最后一步的梯度.所以要么就有$g=0$,意思就是已经找到了一个固定点(stationary point);要么就是$g\perp d $,意味着这一步所在点的位置上局部梯度和搜索方向相互垂直.因此连续起来方向就是正交的,如图8.3(b)所示,这就解释了搜索路径的扭折行为.

降低这种扭折效应的一种简单的启发式方法就是增加一个动量项(momentum term)$\theta_k-\theta_{k-1}$:

$\theta_{k+1}=\theta_k-\eta_kg_k+\mu_k(\theta_k-\theta_{k-1})$(8.12)

上式中的$0\le \mu_k \le 1$控制了动量项的重要程度.在优化领域中,这个方法叫做重球法(heavy ball method ,参考 Bertsekas 1999).

另外一种最小化扭折行为的方法是使用共轭梯度(conjugate gradients)(参考Nocedal and Wright 2006第五章,或者Golub and van Loan 1996,10.2).这个方法是选择形式为$f(\theta)=\theta^TA\theta$的二次形式(quadratic objectives),这是在解线性方程组的时候出现的.不过非线性的共轭梯度就不太受欢迎了.

### 8.3.3 牛顿法

#### 算法8.1 最小化一个严格凸函数的牛顿法

1. 初始化一个$\theta_0$;
2. 对于k=1,2,...等,一直到收敛为止,重复下面步骤:
3.      估算$g_k=\nabla f(\theta_k)$
4.      估算$H_k=\nabla^2 f(\theta_k)$
5.      对$d_k$求解$H_kd_k=-g_k$
6.      使用线性搜索来找到沿着$d_k$方向的步长$\eta_k$
7.      $\theta_{K+1}=\theta_k+\eta_kd_k$


如果把空间曲率(curvature)比如海森矩阵(Hessian)考虑进去,可以推导出更快速的优化方法.这样方法就成了二阶优化方法了(second order optimization metods).如果不考虑曲率的那个就叫做牛顿法(Newton’s algorithm).这是一个迭代算法(iterative algorithm),其中包含了下面形式的更新步骤:
$\theta_{k+1}=\theta_k-\eta_kH_k^{-1}g_k$(8.13)

完整的伪代码如本书算法2所示.

这个算法可以按照下面步骤推导.设构建一个二阶泰勒展开序列来在$\theta_k$附近估计$f(\theta)$:

$f_{quad}(\theta)=f_k+g^T_k(\theta-\theta_k)+\frac{1}{2}(\theta-\theta_k)^TH_k(\theta-\theta_k)$(8.14)

重写成下面的形式

$f_{quad}(\theta)=\theta^TA\theta+b^T\theta+c$(8.15)

其中:

$A=\frac{1}{2}H_k,b=g_k-H_k\theta_k,c=f_k-g^T_k\theta_k+\frac{1}{23}\theta^T_kH_k\theta_k$(8.16)

$f_{quad}$最小值为止在:

$\theta=-\frac{1}{2}A^{-1}b=\theta_k-H_k^{-1}g_k$(8.17)

因此牛顿步长$d_k=-H_k^{-1}g_k$就可以用来加到$\theta_k$上来最小化在$\theta_k$附近对$f$的二阶近似.如图8.4(a)所示.

此处参考原书图8.4

在最简单的形式下,牛顿法需要海森矩阵$H_k$为正定矩阵,这保证了函数是严格凸函数.否则,目标函数非凸函数,那么海森矩阵$H_k$就可能不正定了,所以$d_k=-H_k^{-1}g_k$就可能不是一个下降方向了(如图8.4(b)所示).这种情况下,简单的办法就是逆转最陡下降方向,$d_k=-g_k$.列文伯格-马夸特算法(Levenberg Marquardt algorithm)是一种在牛顿步长和最陡下降步长之间这种的自适应方法.这种方法广泛用于解非线性最小二乘问题.一种替代方法就是:不去直接计算$d_k=-H_k^{-1}g_k$,可以使用共轭梯度来解关于$d_k$的线性方程组$H_kd_k=-g_k$.如果$H_k$不是正定矩阵,只要探测到了负曲率,就可以简单地截断共轭梯度迭代,这样就叫做截断牛顿法(truncated Newton).

### 8.3.4 迭代重加权最小二乘法(Iteratively reweighted least squares,缩写为IRLS)


接下来试试将牛顿法用到二值化逻辑回归中求最大似然估计(MLE)上面.在这个模型中第$k+1$次迭代中牛顿法更新如下所示(设$\eta_k=1$,因此海森矩阵(Hessian)是确定的):

$$
\begin{aligned}
w_{k+1}&=  w_k-H^{-1}g_k &\text{(8.18)}\\
&= w_k+(X^TS_kX)^{-1}X^T(y-\mu_k)   &\text{(8.19)}\\
&= (X^TS_kX)^{-1}[(X^TS_kX)w_k+X^T(y-\mu_k)]  &\text{(8.20)}\\
&= (X^TS_kX)^{-1}X^T[S_kXw_k+y-\mu_k]  &\text{(8.21)}\\
&= (X^TS_kX)^{-1}X^TS_kz_k  &\text{(8.22)}\\
\end{aligned}
$$

然后就可以定义工作响应函数(working response)如下所示:
$z_k\overset{\triangle}{=} Xw_k+S_k^{-1}(y-\mu_k)$(8.23)

等式8.22就是一个加权最小二乘问题(weighted least squares problem),是要对下面的项最小化:


$\sum^N_{i=1}S_{ki}(z_{ki}-w^Tx_i)^2$(8.24)

由于$S_k$是一个对角矩阵,所以可以把目标函数写成成分的形式(对每个$i=1:N$):


$z_{ki}=w_k^Tx_i+\frac{y_i-\mu_{ki}}{\mu_{ki}(1-\mu_{ki})}$(8.25)

这个算法就叫做迭代重加权最小二乘法(Iteratively reweighted least squares,缩写为IRLS),因为每次迭代都解一次加权最小二乘法,其中的权重矩阵$S_k$在每次迭代都变化.伪代码参考本书配套算法10.



#### 算法8.2 迭代重加权最小二乘法(IRLS)

1. $w=0_D$;
2. $w_0=\log(\bar y/ (1-\bar y))$
3. 重复下面步骤:
4.      $\eta_i=w_0+w^Tx_i$
5.      $\mu_i=sigm(\eta_i)$
6.      $s_i=\mu_i(1-\mu_i)$
7.      $z_i=\eta_i+\frac{y_i-\mu_i}{s_i}$
8.      $S=diag(s_{1:N})$
9.      $w=(X^TSX)^{-1}X^TSz$
10.  直到收敛


### 8.3.5 拟牛顿法

二阶优化算法的源头都是牛顿法,在本书8.3.3中讲到过.不过很不幸的是计算出来海森矩阵H的运算开销成本太高了.拟牛顿法(Quasi-Newton methods)就应运而生了,以迭代方式使用从每一步的梯度向量中学到的信息来构建对海森矩阵的估计.最常用的方法就是BFGS方法(这四个字母是发明这个算法的四个人的名字的首字母Broyden, Fletcher, Goldfarb, Shanno),这个方法是使用下面所示定义的$B_k\approx H_k$来对海森矩阵进行估计:
$$
\begin{aligned}
B_{k+1}& =B_k+\frac{y_ky_k^T}{y_k^Ts_k}-\frac{(B_ks_k)(B_ks_k)^T}{s_k^TB_ks_k} &\text{(8.26)}\\
s_k& = \theta_k-\theta_{k-1}  &\text{(8.27)}\\
y_k& = g_k-g_{k-1}  &\text{(8.28)}\\
\end{aligned}
$$

这是对矩阵的二阶更新(rank-two update),这确保了矩阵保持正定(在每个步长的特定限制下).通常使用一个对角线估计来启动算法,即设$B_0=I$.所以BFGS方法可以看做是对海森矩阵使用对角线加上低阶估计的方法.

另外BFGS方法也可以对海森矩阵的逆矩阵进行近似,通过迭代更新 $C_l\approx = H_k^{-1}$,如下所示:

$C_{k+1}=(I-\frac{s_ks_k^T}{y_k^Ts_k})C_k(I- \frac{y_ks_k^T}{y_k^Ts_k})+\frac{s_ks_k^T}{y_k^Ts_k}$(8.29)

存储海森矩阵(Hessian)需要消耗$O(D^2)$的存储空间,所有对于很大规模的问题,可以使用限制内存BFGS算法(limited memory BFGS),缩写为L-BFGS,其中的$H_k$或者$H_k^{-1}$都是用对角矩阵加上低阶矩阵来近似的.具体来说就是积(product)$H_k^{-1}g_k$可以通过一系列的$s_k$和$y_k$的内积来得到,只使用m个最近的$s_k,y_k$对,忽略掉更早的信息.这样存储上就只需要$O(mD)$规模的空间了.通常设置m大约在20左右,就足够有很好的性能表现了.更多相关信息参考(Nocedal and Wright 2006, p177).L-BFGS在机器学习领域中的无约束光滑优化问题中通常都是首选方法.

### 8.3.6 $l_2$规范化(regularization)


相比之下我们更倾向于选岭回归而不是线性回归,类似得到了,对于逻辑回归我们更应该选最大后验估计(MAP)而不是计算最大似然估计(MLE).实际上即便数据规模很大了,在分类背景下规范化还是很重要的.假设数据是线性稀疏的.这时候最大似然估计(MLE)就可以通过$||m||\rightarrow \infty$来得到,对应的就是一个无穷陡峭的S形函数(sigmoid function)$I(w^Tx>w_0)$,这也叫一个线性阈值单元(linear threshold unit).这将训练数据集赋予了最大规模概率质量.不过这样一来求解就很脆弱而且不好泛华.

所以就得喝岭回归里面一样使用$l_2$规范化(regularization).这样一来新的目标函数.梯度函数.海森矩阵就如下所示:

$$
\begin{aligned}
f'(w)&=NLL(w)+\lambda w^Tw           &\text{(8.30)}\\
g'(w)&= g(w)+\lambda w          &\text{(8.31)}\\
H'(w)&= H(w)+\lambda I          &\text{(8.32)}\\
\end{aligned}
$$

调整过之后的等式就更适合用于各种基于梯度的优化器了.

### 8.3.7 多类逻辑回归

接着就要讲多想逻辑回归(multinomial logistic regression)了,也叫作最大熵分类器(maximum entropy classifier).这种方法用到的模型形式为:

$p(y=c|x,W)=\frac{\exp(w_c^Tx)}{\sum^C_{c'=1}\exp(w_{c'}^Tx)}$(8.33)

还有一个轻微变体,叫做条件逻辑模型(conditional logit model),是在对每个数据情况下一系列不同的类集合上进行了规范化;这个可以用于对用户在不同组合的项目集合之间进行的选择进行建模.

然后引入记号.设$\mu_{ic}=p(y_i=c|x_iW)=S(\eta_i)c$,其中的$\eta_iW^Tx_i$是一个$C\times 1$ 向量.然后设$y_{ic}=I(y_i=c)$是对$y_i$的一种编码方式(one-of-C encoding);这样使得$y_i$成为二进制位向量(bit vector),当且仅当$y_i=c$的时候第c个位值为1.接下来是参考(Krishnapuram et al. 2005),设$w_C=0$,这样能保证可辨识性(identifiability),然后定义$w=vec(W(:,1:C-1))$是一个$D\times (C-1)$维度的列向量(column vector).

这样就可以写出如下面形式的对数似然函数(log-likelihood):

$$
\begin{aligned}
l(W)&= \log\prod^N_{i=1}\prod^C_{c=1}\mu_{ic}^{y_{ic}}=\sum^N_{i=1}\sum^C_{c=1}y_{ic}\log \mu_{ic}  &\text{(8.34)}\\
&= \sum^N_{i=1}[(\sum^C_{c=1}y_{ic}w_c^Tx_i)-\log(\sum^C_{c'=1} \exp (w_{c'}^Tx_i) )]  &\text{(8.35)}\\
\end{aligned}
$$

加上负号就是负对数似然函数NLL了:
$f(w)=-l(w)$

然后就可以针对NLL计算器梯度和海森矩阵了.由于w是分块结构的(block-structured),记号会比较麻烦,但思路还是很简单的.可以定义一个$A\otimes B$表示在矩阵A和B之间的克罗内克积(kronecker product).如果A是一个$m\times n$矩阵,B是一个$p\times q$矩阵,那么$A\otimes B$就是一个$mp\times nq$矩阵:

$A\otimes B= \begin{bmatrix} a_{11}B &...& a_{1n}B\\...&...&...\\a_{m1}B&...&a_{mn}B \end{bmatrix}$(8.37)

回到咱们刚刚的例子,很明显(练习8.4)梯度为:
$g(W)=\nabla f(w)=\sum^N_{i=1}(\mu_i-y_i)\otimes x_i$(8.38)

其中的$y_i=(I(y_i=1),...,I(y_i=C-1))$,$\mu_i(W)=[p(y_i=1|x_i,W),...,p(y_i=C-1|x_i,W)]$这两个都是长度为$C-1$的列向量(column vectors).例如如果有特征维度D=3,类别数目C=3,这就成了:

$g(W)=\sum_i\begin{pmatrix}(\mu_{i1}-y_{i1})x_{i1} \\(\mu_{i1}-y_{i1})x_{i2}\\(\mu_{i1}-y_{i1})x_{i3}\\(\mu_{i2}-y_{i2})x_{i1}\\(\mu_{i2}-y_{i2})x_{i2}\\(\mu_{i2}-y_{i2})x_{i3}\end{pmatrix}$(8.39)


也就是说对于每个类c,第c列中权重的导数(the derivative for the weights)为:
$\nabla_{w_c}f(W)=\sum_i(\mu_i{i}-y_{ic})x_i$(8.40)

这就和二值化逻辑回归的情况下形式一样了,名义上就是误差项乘以$x_i$.(其实这是指数族分布的一个通用特征,在本书9.3.2会讲到.)

(参考练习8.4)很容易发现海森矩阵是一个下面所示形式的$D(C-1)\times D(C-1)$分块矩阵:

$H(WW) = \nabla^2 f(w)=\sum^N_{i=1}(diag(\mu_i)-\mu_i\mu_i^T)\otimes (x_ix_i^T) $(8.41)


$$
\begin{aligned}
H(W)&=\sum_i \begin{pmatrix} \mu_{i1}-\mu_{i1}^2 & -\mu_{i1}\mu_{i2}\\ -\mu_{i1}\mu_{i2}& \mu_{i2}-\mu_{i2}^2\end {pmatrix} \otimes \begin{pmatrix} x_{i1}x_{i1}&x_{i1}x_{i2}&x_{i1}x_{i3}\\x_{i2}x_{i1}&x_{i2}x_{i2}&x_{i2}x_{i3}\\x_{i3}x_{i1}&x_{i3}x_{i2}&x_{i3}x_{i3}  \end{pmatrix}               &\text{(8.42)}\\
&=  \sum_i\begin{pmatrix} (\mu_{i1}-\mu_{i1}^2)X_i&-\mu_{i1}\mu_{i2}X_i\\-\mu_{i1}\mu_{i2}X_i&(\mu_{i2}-\mu_{i2}^2)X_i  \end{pmatrix}             &\text{(8.43)}\\
\end{aligned}
$$

其中的$X_i=x_ix_i^T$.也就是分块矩阵中块$c,c'$部分为:

$H_{c,c'}(W)=\sum_i\mu_{ic}(\delta_{c,c'}-\mu_{i,c'})x_ix_i^T$(8.44)

这也是一个正定矩阵,所以有唯一最大似然估计(MLE).

接下来考虑最小化下面这个式子:
$f'(wW\overset{\triangle}{=} -\log [(D|w)-\log p(W)$(8.45)

其中的$p(W)=\prod_cN(w_C|0,V_0)$.这样一来就得到了下面所示的新的目标函数/梯度函数/海森矩阵:
$$
\begin{aligned}
f'(w)&= f(w)+\frac{1}{2}\sum_cw_cV_0^{-1}w_c &\text{(8.46)}\\
g'(w)&= g(W)+V_0^{-1}(\sum_cw_c) &\text{(8.47)}\\
H'(w)&= H(W)+I_C\otimes V_0^{-1} &\text{(8.48)}\\
\end{aligned}
$$

这样就可以传递给任意的基于梯度的优化器来找到最大后验估计(MAP)了.不过要注意这时候的海森矩阵规模是$O ((CD)\times (CD))$,比二值化情况下要多C倍的行和列,所以这时候更适合使用限制内存的L-BFGS方法,而不适合用牛顿法.具体MATLAB代码参考本书配套PMTK3的logregFit.

## 8.4 贝叶斯逻辑回归

对逻辑回归模型,很自然的需求就是计算在参数上的完整后验分布$p(w|D)$.只要我们相对预测指定一个置信区间,这就很有用了.(另外在解决情景匪徒问题(contextual bandit problems)时候也很有用,参考本书5.7.3.1.)
然而很不幸,不像线性回归的时候了,这时候没办法实现这个目的,因为对于逻辑回归来说没有合适的共轭先验.本章这里讲的是一个简单近似;还有一些其他方法,比如马尔可夫链蒙特卡罗(Markov Chain Monte Carlo,缩写为MCMC,本书24.3.3.1),变分推导(variational inference,本书21.8.1.1),
期望传播(expectation propagation,Kuss and Rasmussen 2005)等等.为了记号简单,这里还用二值逻辑回归为例.

### 8.4.1 拉普拉斯近似(Laplace approximation)

本节将对一个后验分布进行高斯近似.假如$\theta\in R^D$.设:
$p(\theta|D)=\frac{1}{Z}e^{-E(\theta)}$(8.49)

其中的$E(\theta)$叫能量函数(energy function),等于未归一化对数后验(unnormalized
log posterior)的负对数,即$E(\theta)=-\log p(\theta,D)$,$Z=p(D)$是归一化常数.然后进行关于众数$\theta^*$(对应最低能量状态)的泰勒级数展开,就得到了:

$E(\theta) \approx E(\theta^*)+(\theta-\theta^*)^Tg+\frac{1}{2}(\theta-\theta^*)^TH(\theta-\theta^*)$(8.50)

其中的g是梯度,H是在众数位置能量函数的海森矩阵:
$g\overset{\triangle}{=} \nabla E(\theta)|_{\theta^*},H\overset{\triangle}{=} \frac{\partial^2E(\theta)}{\partial\theta\partial\theta^T}|_{\theta^*}$(8.51)

由于$\theta^*$是众数,梯度项为零.因此:
$$
\begin{aligned}
\hat p(\theta|D)& \approx \frac{1}{Z}e^{-E(\theta^*)} \exp[-\frac{1}{2}(\theta-\theta^*)^T H(\theta-\theta^*)] &\text{(8.52)}\\
& = N(\theta|\theta^*,H^{-1})&\text{(8.53)}\\
Z=p(D)& \approx \int \hat p(\theta|D)d\theta = e^{-E(\theta^*)}(2\pi)^{D/2}|H|^{-\frac{1}{2}} &\text{(8.54)}\\
\end{aligned}
$$

最后这一行是参考了多元高斯分布的归一化常数.

等式8.54就是对边缘似然函数的拉普拉斯近似.所以等式8.52有时候也叫做对厚颜的拉普拉斯近似.不过在统计学领域,拉普拉斯近似更多指的是一种复杂方法(具体细节参考Rue等,2009).高斯近似通常就足够近似了,因为后验分布随着样本规模增长就越来越高斯化,这个类似中心极限定理.(在物理学领域有一个类似的技术叫做鞍点近似(saddle point approximation).)










