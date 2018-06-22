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





