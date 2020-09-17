# MLAPP-CN
A Chinese Notes of MLAPP，MLAPP 中文笔记项目  https://zhuanlan.zhihu.com/python-kivy



## 书中疑似错误

#### 14页1.3.4.1图像填充倒数第三行

“This is somewhat like masket basket analysis”

这里的 masket 明显是拼写错误，应该是 market，应该改为：

“This is somewhat like market basket analysis”



#### 22页1.4.7过拟合当中第一段最后一句

“Thus using such a model might result in accurate predictions of future outputs.”

 这句话这样就成了“因此用这样的模型对未来输出可能会做出准确预测。”而实际结合上下文应该是说过拟合的模型对未来输出可能不能给出精确预测，所以是缺少了个 not，应该改成：

“Thus using such a model might not result in accurate predictions of future outputs.”


#### 87页3.5.4最后一段第一句

"Figure 3.1 illustrates what happens if we apply this to the binary bag of words dataset used in Figure 3.8."

图3.1和词汇没关系,是猜数字游戏的,推测应该是表3.1.应该改成:

"Table 3.1 illustrates what happens if we apply this to the binary bag of words dataset used in Figure 3.8."


#### 140页公式4.261最右侧
原文错写成了$I\propto \int p(D_x|\mu,\lambda_x)p(\lambda_x|\mu)d\lambda_x  \propto (\bar x -\mu)^2+s_x^2)^{-1}$(4.261)
应该是$I\propto \int p(D_x|\mu,\lambda_x)p(\lambda_x|\mu)d\lambda_x  \propto [(\bar x -\mu)^2+s_x^2]^{-1}$(4.261)

2018年05月17日08:34:22

#### 100页第四章的(4.19)尼玛是空白?

#### 108页第四章的(4.62)公式第三个项目$\gamma_c$后面缺少个等号,推测应该是:$\gamma_c=- \frac{1}{2}\mu_c^T \beta_c+\log \pi_c $

2018年05月22日16:10:47
#### 116页第四章的(4.91)公式,等号右侧的两个分母都是$\sigma_g^2+\sigma_g^2$,推测是笔误,应该是$\sigma_f^2+\sigma_g^2$

2018年06月03日08:31:16
#### 166页第五章的(5.52)公式,应该是$I(\phi) *= -E[(\frac{d\log p(X|\phi)}{d\phi} )^2]$(5.52),而原文错写成了$I(\phi) *= -E[(\frac{d\log p(X|\phi)}{d\phi} )2]$(5.52)


2018年06月04日07:02:50

#### 170页第五章 5.4.4.2 第二段第三行的 $Z _1 = 1$ 明显是错的,应该是 $Z _t= 1$ .


2018年06月11日09:11:45

#### 200页定理6.3.3证明证毕前的最后一句话,$\delta_2$is admissible似乎明显是错误的,应该写$\delta_1$is admissible.这段本来证明的也是$\delta_1$is admissible.


2018年6月20日15:27:06

#### 246页公式8.4前面的说明中,"We have 有$p(y=1)=\frac{1}{ 1+\exp (-w^Tx)}$ and $p(y=1)=\frac{1}{ 1+\exp (+w^Tx)}$",两处都是p(y=1),应该第一个是p(y=-1),第二个是p(y=1).


2018年6月23日06:29:16
#### 253页公式8.37前面的一段中,分明是说克罗内克积的定义,公式前面那句话却写成了$A\times B$,很明显应该是$A\otimes B$.


2018年6月27日07:47:00
#### 273页公式8.104,原文写成了$w^TS_Bw=w^T(\mu_2-\mu_1)(\mu_2-\mu_1)^T w=(m_2-m_1)(m_2-m_1)$,分明应该是$w^TS_Bw=w^T(\mu_2-\mu_1)(\mu_2-\mu_1)^T w=(m_2-m_1)(m_2-m_1)^T$.



#### 746页第二十一章的(21.91)

应为

$$
\mathbb{E}[ x^2 | x \sim \mathcal{N} (\mu , \sigma^2) ] = \mu^2 + \sigma^2
$$

#### 750页第二十一章的(21.127)

应为

$$
\log q(z) = \sum_k \sum_i z_{ik} \log \rho_{ik} + \text{const}
$$
