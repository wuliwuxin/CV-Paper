---
title: Inception v2/BN-Inception
categories: 计算机视觉
tags:
  - 论文
description: Inception-v2结构的改进就是将原来的Inception-v1结构中的5✖️5卷积层进行修改，用两个3✖️3卷积层代替。
date: 2021-09-12 18:48:14
---

# 带你读论文系列之计算机视觉--Inception v2/BN-Inception
![](https://img-blog.csdnimg.cn/9ea3f5ca05c948908b6b86a70241640a.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

我们终其一生，就是要摆脱他人的期待，找到真正的自己。--《无声告白》

## 概述
[论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

[回顾GoogLeNet](https://wuliwuxin.github.io/2021/07/14/googlenet/)

**Inception-v2结构的改进就是将原来的Inception-v1结构中的5✖️5卷积层进行修改，用两个3✖️3卷积层代替**。

Batch Normalization是google在2015提出的深度学习的优化技巧。

它不仅可以加快模型的收敛速度，而且更重要的是在一定程度缓解了深层网络中“梯度弥散”的问题，从而使得训练深层网络模型更加容易和稳定。

神经网络在训练时候，每一层的网络参数会更新，也就是说下一层的输入的分布都在发生变化，这就要求网络的初始化权重不能随意设置，而且学习率也比较低。因此，我们很难使用饱和非线性部分去做网络训练，作者称这种现象为internal covariate shift。

**Batch Normalization的提出，就是要解决在训练过程中，中间层数据分布发生改变的情况**。

因为现在神经网络的训练都是采用min-batch SGD，所以Batch Normalization也是针对一个min-batch做归一化，这也就是Batch Normalization中batch的由来。在神经网络的训练中，如果将输入数据进行白化预处理（均值为0，方差为1，去相关）可以加快收敛速度。

但是白化处理计算量太大，而且不是处处可微的，所以作者做了两个简化处理。一是对每个维度单独计算，二是使用mini-batch来估计估计均值和方差。

**摘要**
1. 数据分布变化导致训练困难；
2. 低学习率，初始化可解决，但训练慢；
3. 数据分布变化现象称为ICS；
4. 提出Batch Normalization层解决问题；
5. Batch Normalization 优点：大学习率不关心初始化，正则项不用Dropout；
6. 成果：ImageNet分类的最佳发布结果：达到4.9%的top-5验证错误（和4.8%的测试错误），超过了人类评估者的准确度。

**BN-Inception网络—关键点**
- Batch Normalization（批归一化）。意义，目前BN已经成为几乎所有卷积神经网络的标配技巧。
- 5×5卷积核→2个3×3卷积核。相同的感受野。
![](https://img-blog.csdnimg.cn/a7532fff0a5b4f059dc02c80486c2c12.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_14,color_FFFFFF,t_70,g_se,x_16)

**BN 的好处**：
- BN 减少了内部协方差，提高了梯度在网络中的流动，加速了网络的训练。
- BN使得可以设置更高的学习速率。
- BN正则了模型。

## 论文详情

 深度学习极大地提高了视觉、语音和许多其他领域的技术水平。随机梯度下降(SGD)已被证明是训练深度网络的有效方法，并且诸如momentum和
Adagrad等SGD变体已被用于实现最先进的性能。SGD优化网络的参数Θ，从而最小化损失。

使用SGD，训练分步进行，在每一步我们考虑mini-batch。mini-batch用于近似损失函数相对于参数的梯度。

**使用mini-batch的的2个好处**：
 1. 稳定，提高精度；
2. 高效，加快速度。

SGD要调参数：
1. Learning Rate
2. 权重初始化

**缺点**：
每层的输入受前面所有层的影响，而前面层微小的的改变都会被放大。

将**Internal Covariate Shift**(**ICS**)定义为训练过程中网络参数的变化引起的网络激活分布的变化。为了改进训练，我们寻求减少内部协变量偏移。通过将层输入的分布固定为训练进度，我们期望提高训练速度。众所周知，如果输入被白化，网络训练收敛得更快——即线性变换为具有零均值和单位方差，并且去相关。由于每一层都观察由下面的各层产生的输入，因此对每一层的输入实现相同的白化将是有利的。通过对每一层的输入进行白化，我们将朝着实现输入的固定分布迈出一步，从而消除内部协变量移位的不良影响。

**已知经验**：
**当输⼊数据为whitened ,即0均值1方差,训练收敛速度会快**。

实现白化(whitened)：
1. 直接改网络
2. 根据激活值，优化参数，使得输出是白化的。

我们可以在每个训练步骤或某个时间间隔考虑白化激活，通过直接修改网络或通过更改优化算法的参数以取决于网络激活值。然而，如果这些修改穿插在优化步骤中，那么梯度下降步骤可能会尝试以需要归一化的方式更新参数被更新，这减少了梯度步骤的影响。

![](https://img-blog.csdnimg.cn/96dba9faa1b345588f9c1c93438684ec.webp)

其中期望和方差是在训练数据集上计算的。即使特征不相关，这种归一化也会加速收。通过以上公式把每层的特征进行白化并且是逐个维度的。简单的标准化特征值，会改变网络的表达能力例如sigmoid，限制在线性区域。
![](https://img-blog.csdnimg.cn/9d09cf258adc4b89998750e072aaceab.webp)

其中gamma和beta是可学习允许模型自动控制是否需要保留原始表征。以上共识是为了保留网络的表征能力。

在批量设置中，每个训练步骤都是基于整个训练集的，我们将使用整个训练集来对激活进行管理。然而，在使用随机优化时，这是不现实的。因此，我们做了第二个简化：由于我们在随机梯度训练中使用mini-batch，每个mini-batch都产生每个激活的平均值和方差的估计值。这样，用于归一化的统计学可以完全参与到梯度反向传播中。请注意，mini-batch的使用是通过计算每个维度的变量而不是联合协方差来实现的；在联合的情况下，由于mini-batch的大小可能小于被白化的激活的数量，因此需要进行调节，导致奇异协方差矩阵。

![](https://img-blog.csdnimg.cn/1754d823bc6b4230b52debf239cdbea0.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

在算法1中详细说明了BN。在该算法中，ϵ是一个为了数值稳定性而加到mini-batch variance上的常数。

但需要注意的是，BN变换并不是独立处理每个训练示例中的激活。相反，BNγ,β(x)取决于训练示例和mini-batch中的其他示例。

![](https://img-blog.csdnimg.cn/ad54c8bec1484510af0c850ffa3ef23b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)
在传统的深度网络中，过高的学习率可能会导致梯度爆炸或消失，以及陷入不良的局部最小值。批量标准化有助于解决这些问题。通过对整个网络的激活进行标准化，它可以防止参数的微小变化放大为梯度激活的较大和次优变化；例如，它可以防止训练陷入非线性的饱和状态。

使用批量归一化进行训练时，可以看到一个训练示例与小批量中的其他示例相结合，并且训练网络不再为给定的训练示例生成确定性值。在我们的实验中，我们发现这种效果有利于网络的泛化。Dropout通常用于减少过拟合，在批量归一化网络中，我们发现它可以被移除或降低强度。

**实验**

![](https://img-blog.csdnimg.cn/92e17b384beb4841af8d745c77f6b297.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

a）使用和不使用批量归一化训练的MNIST网络的测试准确率与训练步骤的数量。Batch Normalization帮助网络更快地训练并获得更高的准确率。

（b,c)在训练过程中，输入分布向典型的sigmoid演变，显示为{15,50,85}的百分位数。批量归一化使分布更加稳定，并减少了内部协变量的转移。

将批量归一化应用Inception网络的新变体(2014)，在ImageNet分类任务上进行训练(2014)。该网络有大量的卷积层和池化层，有一个softmax层以从1000种可能性中预测图像类。卷积层使用ReLU函数作为非线性。与Inception网络的新变体中描述的网络的主要区别在于，5×5卷积层被两个连续的3×3卷积层替换，最多128 个滤波器。网络包含13.6·106个参数，除了最上面的softmax层，没有全连接层。

简单地将批量归一化添加到网络并不能充分利用我们的方法。为此，我们进一步更改了网络及其训练参数，如下：

1. 增大学习率；
2. 移除Dropout；
3. 降低weight decay 降低5倍，减轻限制权重的大小，因为BN允许权重大一些；
4. 更早的学习率下降总共下降6次；

![](https://img-blog.csdnimg.cn/d1c8ff407c4045ccb6b022487cd245ce.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

这个图展示了为什么是6。

5. 移除LRN v1中进入inception之前用了LRN；
6. 彻底shuffle 充当正则；
7. 减少光照变化因为训练次数少，期望模型看到的是真实的样本。

![](https://img-blog.csdnimg.cn/d47982b49bb841389edbcf4d14415175.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

在提供的包含50000个图像的验证集上，与以前的最新技术进行批量标准化初始比较。*根据测试服务器的报告，在ImageNet测试集的100000张图像上，BN初始集成已达到4.82% top-5。

其中BN-Inception Ensemble，则采用多个网络模型集成学习后得到的结果。

为了启用深度网络训练中常用的随机优化方法，对每个小批量执行归一化，并通过归一化参数反向传播梯度。批量归一化每次激活仅添加两个额外参数，并且这样做保留了网络的表示能力。文章提出了一种使用批量归一化网络构建、训练和执行推理的算法。由此产生的网络可以用饱和非线性进行训练，对增加的训练率有更大的容忍度，并且通常不需要Dropout进行正则化。

只加2个参数就保持表征能力。

**优点**：
1. 可训练饱和激活函数；
2. 可⽤⼤学习率；
3. 不需要dropout。

**Inception+bn+优化**
**单模型SOTA
多模型SOTA**

批量归一化的目标是在整个训练过程中实现激活值的稳定分布，在我们的实验中，我们在非线性之前应用它，因为这时匹配第一和第二时刻更有可能产生稳定的分布。相反，将标准化层应用于非线性的输出，这导致了更稀疏的激活。在我们的大规模图像分类实验中，我们没有观察到非线性输入是稀疏的，无论是使用还是不使用批量规范化。

**下一步研究**
1.  BN+RNN，RNN的ICS更严重；
2.  BN+domain adaptation 仅重新计算mean+std 就可以了。

![](https://img-blog.csdnimg.cn/35d0f14538c941a4a0fc471e0760c538.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)






