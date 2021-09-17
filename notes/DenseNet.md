---
title: DenseNet
categories: 计算机视觉
tags:
  - 论文
description: 本文是重新思考认识short path和feature reuse的意义，引入稠密连接思想。
date: 2021-09-17 23:36:20
---

# 带你读论文系列之计算机视觉--DenseNet
![](https://img-blog.csdnimg.cn/285b88c882c443399bf83db7ef1f5df5.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_16,color_FFFFFF,t_70,g_se,x_16)

情若能自控，我定会按捺住我那颗吃货的心。


## 闲谈
今天听了师兄申请博士的经验。第一是感觉历程很心累，压力也很大；二是成功后很喜悦；三是成果很重要，其次是关系，努力和运气。漫长的时间等待与艰辛的的经历。对于现在的我来说，更多的是脚踏实地打好基础，不应该过于急于求成，慢慢来会更快。在一次次的选择后，我需要做到的就是减少自己的后悔。也许每一次的选择并不完美，也有利弊的取舍，收拾好心情又要重新出发。明天太阳🌞升起，又是美好的一天⛽️。

## 引言
**论文**：[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

[**代码**](https://github.com/liuzhuang13/DenseNet)

**本文是重新思考认识short path和feature reuse的意义，引入稠密连接思想。**

DenseNet在ResNet的基础上([回顾ResNet](/notes/ResNet和ResNeXt.md)🐂，ResNet 经典！！！)，进一步扩展网络连接，对于网络的任意一层，该层前面所有层的feature map都是这层的输入，该层的feature map是后面所有层的输入。

**摘要**：

1. 残差连接使CNN更深、更强、更高效；

2. 本文提出DenseNet，特点是前层作为后面所有层的连接；

3. 通常L层有L个连接，DenseNet有L*（L+1）/2 ；
4. 靠前的层作为靠后所有层的输入；
5. DenseNet优点减轻梯度消失，增强特征传播，加强特征复用，减少权重参数；
6. 在4个数据集上进行验证，表现SOTA。


如果将接近输入和输出的层之间短接，卷积神经网络可以更深、精度更高且高效。最后提出了密集卷积网络(DenseNet)，它的每一层在前向反馈模式中都和后面的层有连接，与L层传统卷积神经网络有L个连接不同，DenseNet中每个层都和其之后的层有连接，因此L层的DenseNet有 L(L+1)/2 个连接关系。
 
在四个目标识别的基准测试集（CIFAR-10、CIFAR-100、SVHN 和 ImageNet）上评估了我们的结构，可以发现DenseNet在减少计算量的同时取得了更好的表现。


## 论文详情

稠密连接（Dense connectivity）

![](https://img-blog.csdnimg.cn/189e3d35498f409d8f4c4e5e94557bca.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_15,color_FFFFFF,t_70,g_se,x_16)

说明了生成的DenseNet的布局。在每一个block中，每一层的输出均会直接连接到后面所有层的输入。为了确保网络中各层之间的最大信息流，我们将所有层（具有匹配的特征图大小）直接相互连接。为了保持前馈性质，每一层从所有前面的层获得额外的输入，并将它自己的特征图传递给所有后续层。
 
ResNets和Highway Networks通过身份连接将信号从一层旁路到下一层。随机深度通过在训练期间随机删除层来缩短ResNet，以允许更好的信息和梯度流。
 
为了确保网络中各层之间的最大信息流，我们将所有层（具有匹配的特征图大小）直接相互连接。为了保持前馈性质，每一层从所有前面的层获得额外的输入，并将它自己的特征图传递给所有后续层。
 
**最重要的是，与ResNet不同的是，我们在将特征传递到一个层之前，从不通过求和的方式将它们结合起来；相反，我们通过串联的方式来结合特征。**
 
1. 稠密连接可用更少参数获得更多特征图；
2. 传统NN的层可看成一个状态，NN对状态进行修改或保留；
3. ResNet通过恒等映射保留信息；
4. ResNet中很多特征图作用很小；
5. ResNet的状态（特征图）就像RNN网络一样；
6. DenseNet不进行特征相加以及信息保留；
7. DenseNet网络很窄。
 
Highway Network 与ResNet的成功均得益于By passing paths。
 
1. 除了加深网络，还有一种加宽的方法来提升网络；
2. 如GoogLenet ，加宽的ResNet。

![](https://img-blog.csdnimg.cn/d947b1d4057e4dc89af508faa6c0b172.webp)

在深度足够的情况下，ResNet可以提高其性能。FractalNet也在几个数据集上使用广泛的网络结构取得了有竞争力的结果。
 
 - DenseNet不同于其他网络去增加宽度和深度;
 - DenseNet利用特征复用，得到易训练且高效的网络。
 
DenseNet不是从极深或极广的架构中获取表征能力，而是通过特征重用来发挥网络的潜力，产生易于训练且参数效率高的浓缩模型。将不同层学到的特征图串联起来，增加了后续层输入的变化，提高了效率。这构成了DenseNet 和ResNet的一个主要区别。Inception networks也是将不同层的特征连接起来，与之相比，DenseNet更简单、更有效。
 
还有其他值得注意的网络结构创新，它们产生了有竞争力的结果。网络中的网络（NIN）结构将微型多层感知器纳入卷积层的过滤器，以提取更复杂的特征。在Deeply Supervised Net-work(DSN)中，内部各层直接受到辅助分类器的监督，这可以加强早期各层所接收的梯度。梯形网络在自动编码器中引入了横向连接，在半监督学习任务中产生了令人印象深刻的精确度。深度融合网（DFN）被提出，通过结合不同基础网络的中间层来改善信息流。

 - shortcut connection的更容易优化；
 - 缺点是求和的形式会阻碍（impede）信息的流通；

- CNN需要下采样但Denseblock中分辨率是不会变的
- 在block之间进行特征图分辨率下降
- 利用transition layer来执行
- BN+1*1conv+2*2 池化
 
 
1. 若每层计算后得到k个特征图，那么第l 层会有k0+k*(l-1)个特征
图，因此k不能太大；
2. DenseNet的每层就非常的窄，非常的薄，例如k=12；
3. 这里的k就是超参数Growth Rat；
4. k越小，结果越好；
5. 从state的角度讲解DenseNet；

这篇文章又一个比较清晰的bottleneck解释。
 
每个3×3卷积之前引入1×1卷积作为bottleneck，以减少输入特征图的数量，从而提高计算效率。

为了进一步提高模型的紧凑性，我们可以减少过渡层的特征图的数量。如果一个密集块包含m个特征图，我们让下面的过渡层生成bθmc输出特征图，其中0<θ≤1称为压缩因子。当θ=1时，跨过渡层的特征图数量保持不变。我们将θ<1的DenseNet称为DenseNet-C，我们在实验中设置θ=0.5。当同时使用θ<1的Bottleneck和过渡层时，我们将我们的模型称为DenseNet-BC。
 
在除ImageNet之外的所有数据集上，我们实验中使用的DenseNet具有三个密集块，每个块都有相同的层数。在进入第一个密集块之前，对输入图像执行具有16 个（或DenseNet-BC增长率的两倍）输出通道的卷积。对于内核大小为3×3的卷积层，输入的每一侧都填充了一个像素以保持特征图大小固定。我们使用1×1 卷积，然后是2×2平均池化作为两个连续密集块之间的过渡层。在最后一个密集块的末尾，进行全局平均池化，然后附加一个softmax 分类器。三个密集块中的特征图大小分别为32×32、16×16和8×8。我们用配置{L=40,k=12}、{L=100,k =12}和{L=100,k=24}对基本的DenseNet结构进行实验。对于DenseNet-BC，评估具有配置{L=100,k=12},{L=250,k=24}和{L=190,k=40}的网络。

在我们在ImageNet上的实验中，我们在224×224 输入图像上使用具有4个密集块的DenseNet-BC结构。初始卷积层包括2k个大小为7×7、步幅为2的卷积；所有其他层中特征图的数量也来自设置k。我们在ImageNet上使用的确切网络配置如下图所示：
![](https://img-blog.csdnimg.cn/518c202e9b7a40a7bd42d7711e0144ff.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

DenseNet-121是指网络总共有121层：(6+12+24+16)*2 + 3(transition layer) + 1(7x7 Conv) + 1(Classification layer) = 121。
 
过渡层；
1×1卷积，然后是2×2平均池化。


## 实验

![](https://img-blog.csdnimg.cn/22669fc3916a4a8d85cd66a9820c3da9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

L表示网络深度，k为growth rate。蓝色字体表示最优结果，+表示对原数据集进行data augmentation。DenseNet相比ResNet取得更低的错误率，且参数更少。
 
**讨论DenseNet的优点**
![](https://img-blog.csdnimg.cn/c93aad26941d4d568f7870b729bf34d9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)
![](https://img-blog.csdnimg.cn/efaa94bbd3ff4a87b5f5347a7e69fc6c.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_19,color_FFFFFF,t_70,g_se,x_16)

结论：加B，加C更省参数。

![](https://img-blog.csdnimg.cn/a6568b3368f2497b8b5f39e2ef3fa06b.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_15,color_FFFFFF,t_70,g_se,x_16)

同样精度时，densenet-bc仅用1/3参数。

![](https://img-blog.csdnimg.cn/2e3fc616fe24446dbace6e5145ca08d5.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)


同样精度，相差十几倍的参数量。

将输入进行连接的直接结果是，DenseNets每一层学到的特征图都可以被以后的任一层利用。该方式有助于网络特征的重复利用，也因此得到了更简化的模型。DenseNet-BC仅仅用了大概ResNets 1/3的参数量就获得了相近的准确率

三个dense block的热度图如下图所示：

![](https://img-blog.csdnimg.cn/7a8248450c7843e9a6c1cc7b8598ff1b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

在一个训练有素的密集网络中，卷积层的平均绝对滤波权重。像素(s)的颜色编码了连接卷积层和`密集区的权重的平均水平（按输入特征图的数量归一化）。由黑色矩形突出的三列对应于两个过渡层和分类层。第一行是连接到密集块输入层的权重。

**可以得出以下结论**

1. 在同一个block中，所有层都将它的权重传递给其他层作为输入。这表明早期层提取的特征可以被同一个dense block下深层所利用；

2. 过渡层的权重也可以传递给之前dense block的所有层，也就是说DenseNet的信息可以以很少的间接方式从第一层流向最后一层；

3. 第二个和第三个dense block内的所有层分配最少的权重给过渡层的输出，表明过渡层输出很多冗余特征。这和DenseNet-BC强大的结果有关系；

4. 尽管最后的分类器也使用通过整个dense block的权重，但似乎更关注最后的特征图，表明网络的最后也会产生一些高层次的特征。


## 总结

我们提出了一种新的卷积网络架构，我们称之为密集卷积网络（DenseNet ）。它引入了具有相同特征图大小的任意两层之间的直接连接。我们展示了DenseNet。自然地扩展到数百层，同时没有表现出优化困难。

随着参数数量的增加，DenseNets倾向于在准确度上持续提高，而没有任何性能下降或过度拟合的迹象。在多个设置下，它在几个竞争激烈的数据集上取得了最先进的结果。此外，DenseNets需要更少的参数和更少的计算来实现最先进的性能。因为在我们的研究中我们采用了针对残差网络优化的超参数设置，我们相信可以通过更详细地调整超参数和学习率计划来进一步提高
DenseNets的准确性。

在遵循简单的连接规则的同时，DenseNets自然地集成了身份映射、深度监督和多样化深度的特性。它们允许在整个网络中重复使用特征，因此可以学习更紧凑，更准确的模型。由于其紧凑的内部表示和减少的特征冗余，DenseNets 可能是各种基于卷积特征的计算机视觉任务的良好特征提取器。在未来的工作中用DenseNets 研究这种特征转移。

![](https://img-blog.csdnimg.cn/8e8fc1bac2f44b6a8f0c348016c1b945.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)
