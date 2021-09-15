---
title: Inception-V4
categories: 计算机视觉
tags:
  - 论文
description: CNN很强，例如我们的Inception；最近的resnet也很强。那强强联手会怎么样呢？
date: 2021-09-15 17:08:30
---
# 带你读论文系列之计算机视觉--Inception V4
![](https://img-blog.csdnimg.cn/5260b9a877e745a795283e4ce9360233.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

 一直在路上,不是在奔跑,就是在漫步。
 
##  前言

**论文**：
**[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)**

**CNN很强，例如我们的Inception；最近的resnet也很强。那强强联手会怎么样呢**？

1. 速度方面：残差学习可加快inception收敛；
2. 精度方面：残差学习仅带来一小部分提升；
3. 提出新模型；
4. 提出激活值缩放技巧来训练模型；
5. 在2015ILSVRC挑战中取得了 SOTA(State Of The Art), 它的性能类似于最新一代的Inception-v3 网络；
6. 使用三个残差和一个Inception-v4的集合，在ImageNet classification (CLS)挑战的测试集上实现3.08%的top-5 错误。

**残差Inception网络在没有残差连接的情况下比同样的Inception网络表现出色。**

[回顾Res Net](/notes/ResNet和ResNeXt.md)

主要思想就是将residual和inception结构相结合，以获得residual带来的好处。

## 论文详情
由于Inception网络往往非常深，因此用残差连接替换Inception架构的过滤器级联阶段。

**ResNet网络的亮点**：
- 超神的网络结构（突破1000层）；
- 提出Residual模块；
- 使用Batch Normalization加速训练（丢弃Dropput）；

**因此Inception获得残差方法的所有好处，同时保持其计算效率。**

[带你读论文系列之计算机视觉--GoogLeNet](/notes/GoogLeNet.md)

[带你读论文系列之计算机视觉--Inception v2/BN-Inception](/notes/Inception-v2-BN-Inception.md)

[带你读论文系列之计算机视觉--GoogLeNet V3](/notes/GoogLeNet-V3.md)

模型的参数和计算复杂性束缚了Inception V3 的性能。而Inception V4它具有比Inception V3更统一的简化架构和更多的Inception模块。

Inception V4和Inception-ResNet V2的表现相似，超过了最先进的单帧性能ImageNet验证集。

![](https://img-blog.csdnimg.cn/8c352b66ae574ea6997a32418089f249.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_14,color_FFFFFF,t_70,g_se,x_16)

残差模块的引入。

![](https://img-blog.csdnimg.cn/388a6a337a6f4bdfb45f579ef3384bc9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

ResNet的残差连接。

**v4要针对v3进行一系列简洁的优化。**

Inception的结构是很容易调节的，就是说改变一些fitler最终并不会影响结果。但是作者为了优化训练速度小心的调整了每一层的大小。现在因为tensorflow的优化功能，作者认为不需要再像以前一样根据经验小心的调整每一层，现在可以更加标准化的设置每一层的参数。而提出了Inception-V4，网络结构如下：

![](https://img-blog.csdnimg.cn/0fe0a191ffba431ab00c058e28027a43.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

![](https://img-blog.csdnimg.cn/9229d3cfe34a4b17a80a44d2e1355ab3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_15,color_FFFFFF,t_70,g_se,x_16)

纯Inception-v4和Inception-ResNet-v2网络的Stem模式。这是这些网络的输入部分。

Inception V4具体的Block如下图：

![](https://img-blog.csdnimg.cn/7cd1669ac43140adb37e828946a208c2.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

纯Inception-v4网络的35×35网格模块的模式。这是Inception-V4网络结构中的Inception-A块。

![](https://img-blog.csdnimg.cn/c6cfa7565be2410592e67776c0bf0394.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

纯Inception-v4网络的17×17grid模块的模式。这是Inception-V4网络结构中的Inception-B块。

![](https://img-blog.csdnimg.cn/97fe75c5d9eb4172bb63165101a85b7d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

纯Inception-v4网络的8×8 grid模块模式。这是Inception-V4网络结构中的Inception- C块。

![](https://img-blog.csdnimg.cn/8028063636934dbdbd59b34ac0e41336.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

35×35到17×17reduction模块的模式。

![](https://img-blog.csdnimg.cn/e88ee0ae1947417daef6e678991b6e31.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

17×17到8×8grid-reduction模块的模式。这是Inception-V4网络结构中的纯Inception-v4 network使用的reduction模块。

![](https://img-blog.csdnimg.cn/2334e63cdb0a4611b7d2d073705b79b8.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_19,color_FFFFFF,t_70,g_se,x_16)


Inception-ResNet-v1网络的35×35网格（Inception-ResNet-A）模块的模式。


![](https://img-blog.csdnimg.cn/66b28ae77a0d48c8a6277b89cab221eb.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)


Inception-ResNet-v1网络的17×17 grid（Inception-ResNet-B）模块的模式。

![](https://img-blog.csdnimg.cn/b070d1c0dd4a4d6b8948240176487e52.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

较小的Inception-ResNet-v1网络所使用的这个模块是"Reduction-B"17×17至8×8 grid还原模块。

![](https://img-blog.csdnimg.cn/1718910f4adf4a95bdaa35efd588221b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

Inception-ResNet-v1网络的8×8网格（Inception-ResNet-C）模块的模式。

![](https://img-blog.csdnimg.cn/040f500ec6a64bb5bede28cffaf9a625.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

Inception-ResNet-v1网络的主干。

![](https://img-blog.csdnimg.cn/c9aba3119cc44454b9b682df4c88f2e4.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

Inception-ResNet-v1和Inception-ResNet-v2网络的架构。此架构适用于两个网络，但底层组件不同。

Inception-ResNet-v2具体的Block如下图：

![](https://img-blog.csdnimg.cn/e0c325dd2be6417fbb89f10f88c5c7d4.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

Inception-ResNet-v2网络的35×35grid（Inception-ResNet-A）模块的模式。

![](https://img-blog.csdnimg.cn/6990041aa92840478b9e745b3a2c2e6c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

Inception-ResNet-v2网络的17×17grid（Inception-ResNet-B）模块的模式。

![](https://img-blog.csdnimg.cn/a3b4333448de4e9c9dd5597c95eb26bc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

17×17至8×8网格还原模块的模式。图中更广泛的Inception-ResNet-v1network所使用的Reduction-B模块。

![](https://img-blog.csdnimg.cn/93c8c8ebc8d04946b622a197031f34f1.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

Inception-ResNet-v2网络的8×8 grid（Inception-ResNet-C）模块的模式。

Inception-V4和Inception-Resnet-V2的总体结构是比较像的，都是一个stem加上多次重复的Inception或者Inception-Resnet block，然后后面再连接reduction，然后重复这样的结构几次。

![](https://img-blog.csdnimg.cn/d8eb8f95845e4dcbb651bc0011522164.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

三种Inception变体的Reduction-A模块的filter数量。

![](https://img-blog.csdnimg.cn/5e89539eaba34813b5a83995edb77a62.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_19,color_FFFFFF,t_70,g_se,x_16)

K代表1✖️1 Conv，l代表3✖️3 Conv，m代表 3✖️3 Conv stride为2，n代表 3✖️3 Conv stride为2。

![](https://img-blog.csdnimg.cn/d6e1c86c64b24a0c8726e9077949a1f1.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

当超过1000个卷积核时，会出现“死”神经元。在最后的平均。池化层之前会出现输出值为0 的现象。解决方案是要么减小learning rate，要么对这些层增加额外的batch normalization。

如果将残差部分缩放后再跟需要相加的层相加，会使网络在训练过程中更稳定。因此缩放块只是用一个合适的常数来缩放最后的线性激活，通常在0.1 左右，用这个缩放因子去缩放残差网络，然后再做加法。求和前进行缩放，可稳定训练。缩放系数为0.1-0.3之间。

类似的不稳定在resnet中也有resnet提出warm-up来解决。当卷积核很多，很小的学习率（0.00001）也不能让训练稳定。

**scaling并不是必须的！是否找到某种应用场景，让scaling成为必须的，此为一个可研究方向。**


### 实验
![](https://img-blog.csdnimg.cn/067feff851044e58858f708452326df6.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

Inception-v3训练期间的TOP-1error与计算成本相似的残余Inception相比。评估是在ILSVRC-2012验证集的非黑名单图像的单一作物上进行的。**Residual version 的训练速度要快得多，最终准确率也比传统的Inception-v4略高**。

![](https://img-blog.csdnimg.cn/8dec779cf06a4dc5b885d11262f57aba.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

Single crop-Single model的实验结果。报告了ILSVRC2012验证集的非黑名单子集。


![](https://img-blog.csdnimg.cn/ff3548b233dc467cbc153aefdb1ef8d2.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)


可以看出Inception-V4和Inception-ResNet-V2的差别并不大，但是都比Inception-V3和Inception-ResNet-V1都好很多。

 ### 总结

1.  **Inception-ResNet-v1**：一个混合的Inception版本
2.  **Inception-ResNet-v2**：一个成本较高的混合Inception版本，其识别能力显著提高。
3.  **Inception-V4**：纯粹的Inception变体，无残余连接，其识别能力与Inception-ResNet-v2 大致相同。

主要研究了如何用residual learning 来提升inception的训练速度（紧扣主题，residual learning 只能加快训练，对精度提升没什么用）。此外，我们最新的模型（有和没有残差连接）优于我们以前的所有网络，仅因为模型尺寸的增加。

![](https://img-blog.csdnimg.cn/634f90ba54414607b8c292b48ab3df77.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)









