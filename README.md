# MLAPP-CN
A Chinese Notes of MLAPP，MLAPP 中文笔记项目  https://zhuanlan.zhihu.com/python-kivy


# 本项目不支持在线浏览器直接读取
原则上将不再提供Markdown文本之外的任何其他格式的文件，在校验之后可能会不定期更新pdf，但是就别指望了。
还是自己clone到本地想办法解决阅读问题吧，不愿意动手探索就没办法了。
在VS Code中使用[mdmath](https://github.com/goessner/mdmath)这个插件，就可以实时预览Markdown当中的latex了，也可以去[这个链接]( https://marketplace.visualstudio.com/items?itemName=goessner.mdmath)直接安装.

# 各章节HTML链接




[00 PreFace](https://htmlpreview.github.io/?https://github.com/Kivy-CN/MLAPP-CN/master/HTML/00%20PreFace.html)

[01 Introduction](https://htmlpreview.github.io/?https://github.com/Kivy-CN/MLAPP-CN/master/HTML/01%20Introduction.html)
	

[02 Probability](https://htmlpreview.github.io/?https://github.com/Kivy-CN/MLAPP-CN/master/HTML/02%20Probability.html)

[03 Generative models for discrete data](https://htmlpreview.github.io/?https://github.com/Kivy-CN/MLAPP-CN/master/HTML/03%20Generative%20models%20for%20discrete%20data.html)

[04 Gaussian models](https://htmlpreview.github.io/?https://github.com/Kivy-CN/MLAPP-CN/master/HTML/04%20Gaussian%20models.html)

[05 Bayesian statistics](https://htmlpreview.github.io/?https://github.com/Kivy-CN/MLAPP-CN/master/HTML/05%20Bayesian%20statistics.html)

[06 Frequentist statistics](https://htmlpreview.github.io/?https://github.com/Kivy-CN/MLAPP-CN/master/HTML/06%20Frequentist%20statistics.html)



# 笔记项目概述
本系列是一个新坑，之前读到了Christopher M. Bishop的 《PRML》（Pattern Recognition and Machine Learning） 和 周志华老师的《机器学习》（西瓜书），但由于我自身水平所限，接受起来还挺困难。**（但是老子不是没读！）**

后来看到了 Kevin P. Murphy 的《MLAPP》（Machine Learning A Probabilistic Perspective）感觉其中的篇章结构和公式感受都很对胃口。
本来想翻译一下的，但马上意识到自己水平差太远，根本没能力保障这种鸿篇巨制的翻译质量。毕竟之前简单的 [CS229 课件讲义](https://zhuanlan.zhihu.com/p/28702753) 翻译质量都让我很不满意。而我又不太会使用谷歌翻译和塔多思之类的辅助工具，一直都是凭借笨脑袋来笨翻译。**（其实是怕版权问题！）**
因此我就只分享一下自己的读书笔记，一来简明扼要，二来也能保障写得内容是自己所理解的部分好不至于出太大太荒诞的错误。
另外书中的代码部分，我准备尝试用 Python（3.5 以及之后的版本）来实现一下，也放在 Github 上面: https://github.com/Kivy-CN/MLAPP-CN 。
到时候还希望大家批评指正！








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

## 翻译进度追踪

- [x] 01 Introduction 1~26
- [x] 02 Probability 27~64 (练习略)
- [x] 03 Generative models for discrete data 65~96(练习略)
- [x] 04 Gaussian models 97~148(练习略)
- [x] 05 Bayesian statistics 149~190(练习略)
- [ ] 06 Frequentist statistics 191~216(当前页面200)
- [ ] 07 Linear regression 217~244
- [ ] 08 Logistic regression 245~280
- [ ] 09 Generalized linear models and the exponential family 281~306
- [ ] 10 Directed graphical models (Bayes nets) 307~336
- [ ] 11 Mixture models and the EM algorithm 337~380
- [ ] 12 Latent linear models 381~420
- [ ] 13 Sparse linear models 421~478
- [ ] 14 Kernels 479~514
- [ ] 15 Gaussian processes 515~542
- [ ] 16 Adaptive basis function models 543~588
- [ ] 17 Markov and hidden Markov models 589~630
- [ ] 18 State space models 631~660
- [ ] 19 Undirected graphical models (Markov random fields) 661~706
- [ ] 20 Exact inference for graphical models 707~730
- [ ] 21 Variational inference 731~766
- [ ] 22 More variational inference 767~814
- [ ] 23 Monte Carlo inference 815~836
- [ ] 24 Markov chain Monte Carlo (MCMC) inference 837~874
- [ ] 25 Clustering 875~906
- [ ] 26 Graphical model structure learning 907~944
- [ ] 27 Latent variable models for discrete data 945~994
- [ ] 28 Deep learning 995~1009

