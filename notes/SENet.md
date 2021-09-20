---
title: SENet
categories: 计算机视觉
tags:
  - 论文
description: 让我们的神经网络使用全局信息来增强有用的信息，同时抑制无用的信息。
date: 2021-09-21 00:15:19
---
# 带你读论文系列之计算机视觉--SENet
![](https://img-blog.csdnimg.cn/484b3d0598274f649859636c5d8360db.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)
 
 ## 闲谈

总有那么瞬间思念远方的故人。八月十五中秋节，让我们放下繁忙工作，回家与老人团圆举杯共餐。这是我第一次没有在家过中秋，感觉也还行。现在节日没有什么节日气氛，最重要的家人团聚。各位小可爱们，中秋佳节，愿你快快乐乐，开开心心；健健康康，轻轻松松；团团圆圆，恩恩爱爱；和和美美，红红火火！❤️

## 前言

**[论文：Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)**

**[代码](https://github.com/hujie-frank/SENet)**

一个可嫁接/整合的Block 😇

Momenta在ImageNet2017挑战赛中夺冠的网络架构SENet。本文作者为Momenta高级研发工程师胡杰。

![](https://img-blog.csdnimg.cn/ef6dea9093a14374b523b55812927da1.jpg)

Momenta成立于2016年，是**自动驾驶公司**。其核心技术是基于深度学习的环境感知、高精度地图、驾驶决策算法。产品包括不同级别的自动驾驶方案，以及衍生出的大数据服务。Momenta专注于“打造自动驾驶大脑”，拥有世界专业的深度学习专家，如图像识别领域框架Faster R-CNN和ResNet的作者， ImageNet 2015、ImageNet 2017、MS COCO Challenge 2015等多项比赛。团队成员主要来源于清华大学、麻省理工学院、微软亚洲研究院等高校及研究机构，以及百度、阿里、腾讯、华为、商汤等知名高科技公司，拥有深厚的技术积累、极强的技术原创力和丰富的行业经验。

![](https://img-blog.csdnimg.cn/91909f8d91df4396b95de758eec80cc7.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

SENet获得了ImageNet2017大赛分类任务的冠军，这也是最后一届ImageNet比赛，论文同时获得了CVPR2018的oral。而且，SENet思路简单，实现方便，计算量小，模块化涉及，可以无缝嵌入主流的网络结构中，实践不断证明其可以使得网络获得更好的任务效果。

卷积核作为卷积神经网络的核心，通常被看做是在局部感受野上，将空间上（spatial）的信息和特征维度上（channel-wise）的信息进行聚合的信息聚合体。卷积神经网络由一系列卷积层、非线性层和下采样层构成，这样它们能够从全局感受野上去捕获图像的特征来进行图像的描述。

**摘要**
1. 卷积操作是CNN核心其可融合空间和通道的特征；
2. 已经有人研究增强空间特征的提取；
3. 本文针对通道特征提出SEblock，其可自适应的校正通道特征；
4. SEblock可堆叠成SENet，并在多个数据集上获得较好的效果；
5. SENet仅增加少量参数，就大幅提升精度；
6. 获得ILSVRC冠军；

对ImageNet数据集进行了广泛评估。SENets不局限于某个特定的数据集或任务。通过利用SENets，我们在ILSVRC2017分类竞赛中排名第一。我们的最佳模型集合在测试集上实现了2.251%的最高5级错误1。与前一年的冠军作品相比，这代表了大约25%的相对改进（前五名的误差为2.991%）。

从通道维度入手，设计SEBlock。提出一种机制可对特征进行校正，校正后的特征可保留有价值的特征，剔除没价值的特征，即注意力机制。

![](https://img-blog.csdnimg.cn/7fa8d4911eee4ff6b552da5cf06f1f53.webp)

SE构建块的结构如上图所示。特征首先通过Squeeze操作，它通过在其空间维度（H×W）上聚合特征图来产生通道描述符。该描述符的功能是生成通道特征响应的全局分布的嵌入，允许来自网络的全局感受野的信息被其所有层使用。聚合之后是激励操作，它采用简单的self-gating mechanism的形式，将嵌入作为输入并产生每通道调制权重的集合。这些权重应用于特征映射U以生成SE块的输出，该输出可以直接馈入网络的后续层。可以通过简单地堆叠SE块的集合来构建SE网络(SENet)。此外，这些SE块还可以用作网络架构中一定深度范围内原始块。

## 论文详情

思路：让我们的神经网络使用全局信息来**增强**有用的信息，同时**抑制**无用的信息。

假设：

![](https://img-blog.csdnimg.cn/83a0f10804c8494da40c9bcc705704eb.png)

其中，

![](https://img-blog.csdnimg.cn/ab5d2fd0e8a7479b82ced746b1513ee5.webp)

令K = [K1，K2,.....KC]，其中每个元素Ki为filter kernel
于是：

![](https://img-blog.csdnimg.cn/8ce688fc37f741abbab3396c14a21b8a.webp)

其中*代表了conv运算（忽略bias）

**Squeeze阶段**：

![](https://img-blog.csdnimg.cn/74c2475d20b544ef8a31ec13790a49ed.webp)

**Excitation阶段**：
![](https://img-blog.csdnimg.cn/00c904406c6049f5a3ade629be8c71a2.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

VGGNets和Inception模型表明，增加网络的深度可以显着提高其能够学习的表示质量。通过调节每层输入的分布，批量归一化(BN)为深度网络中的学习过程增加了稳定性，并产生了更平滑的优化表面。在这些工作的基础上，ResNets证明了用shortcut connection来学习更深入、更强大的网络是可能的。Highway Networks引入了一种self-gating machine来调节信息流捷径连接。在这些工作之后，网络层之间的连接有了进一步的重构，其中显示了对深度网络的学习和表示特性的有希望的改进。

1. 分组卷积：ResNeXt
2. 多分支网络GoogLeNet系列
3. 1*1卷积的应用：Xception 等

以往的研究通道之间关系时，采用的是局部信息。本论文提出的方法采用全局方法。

设计和开发新的CNN 架构是一项困难的工程任务，通常需要选择许多新的超参数和层配置。相比之下，SE块的结构很简单，可以直接用于现有的最先进的架构中，通过用SE对应的组件替换，可以有效提高性能。SE模块在计算上也是轻量级的，只在模型复杂性和计算负担上有轻微增加。

**SENet优势**：
1. SE block设计简单，即插即用;
2. SE block参数少

Google团队提出MnasNet（[MnasNet:Platform-AwareNeuralArchitectureSearchforMobile](http://arxiv.org/pdf/1807.11626.pdf)
）使用强化学习的思路，提出一种资源约束的终端CNN模型的自动神经结构搜索方法。MnasNet中用了SEblock。

1. 注意力机制可理解为将最有意义的部分给予更多“关注”；
2. 注意力机制已在序列学习图像理解、定位、图像描述、唇语识别任务中广泛应用；
3. 本论文的block 则是针对通道维度进行注意力机制。

![](https://img-blog.csdnimg.cn/9c179586c6434219968ed96c2ebf8655.webp)

首先是 Squeeze 操作，我们顺着空间维度来进行特征压缩，将每个二维的特征通道变成一个实数，这个实数某种程度上具有全局的感受野，并且输出的维度和输入的特征通道数相匹配。它表征着在特征通道上响应的全局分布，而且使得靠近输入的层也可以获得全局的感受野，这一点在很多任务中都是非常有用的。

其次是 Excitation 操作，它是一个类似于循环神经网络中门的机制。通过参数 w 来为每个特征通道生成权重，其中参数 w 被学习用来显式地建模特征通道间的相关性。

最后是一个 Reweight 的操作，我们将 Excitation 的输出的权重看做是经过特征选择后的每个特征通道的重要性，然后通过乘法逐通道加权到先前的特征上，完成在通道维度上的对原始特征的重标定。

SE网络可以通过简单地堆叠SE构件块的集合来生成。SE块也可以用作体系结构中任何深度的原始块的直接替换。但是，虽然构建模块的模板是通用的，它在不同深度处的角色适应网络的需求。在早期层中，它学会以类不可知的方式激发信息特性，支持共享的底层表示的质量。在后面的层次中，SE块变得越来越专业化，并以 highly class-speciﬁc的方式响应不同的输入。因此，SE块进行特征重新校准的好处可以通过整个网络进行累加。SE块的设计很简单，可以直接与现有最先进的体系结构一起使用，这些体系结构的模块可以通过直接替换SE模块来加强。

![](https://img-blog.csdnimg.cn/6115c7639cf449908274c0db1255c302.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

用公式描述conv2d过程，并且将卷积核按通道维度来理解。

conv2d操作将空间信息与通道信息混合到了一起。本文目的是提高通道维度上信息的敏感度，具体操作为Squeeze和excitation。

![](https://img-blog.csdnimg.cn/5ad66bafc9f74dd09fbd457e8555668b.webp)

**提出问题**：U没有很好的利用局部感受野之外的上下文信息。
**解决问题**：利用全局池化，将空间信息压缩为通道描述符，即数据变为通道维度的形式。该操作可看为图像的局部描述算子，这样的操作在特征工程常见。

1. 为了获取通道之间的信息，加入Excitation；
2. 为实现该目标，需要遵循两个准则：
（1）该操作要能学习通道之间的非线性关系；
（2）确保多个通道能够被“强调”；
3. 采用sigmoid 机制来实现；

![](https://img-blog.csdnimg.cn/eeeaafdefa2e4af8bf1871555806d585.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

上图为激活函数挑选的实验。结论：sigmoid最好。

与其他架构进行整合。论文中把ResNet和Inception进行整合。如下两张图所示。

![](https://img-blog.csdnimg.cn/f4d9a1cc2dbd45b88a9e237d6650b879.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_18,color_FFFFFF,t_70,g_se,x_16)

原始Inception模块（左）和SE-Inception模块（右）的架构。

![](https://img-blog.csdnimg.cn/9fb1eb5c9bd3430fab265324b0972431.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_16,color_FFFFFF,t_70,g_se,x_16)

原始Residual模块（左）和SE-ResNet模块（右）的模式。

![](https://img-blog.csdnimg.cn/1d253d3575fa49d4823a072c6f93e7a2.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_13,color_FFFFFF,t_70,g_se,x_16)

作为这种轻微的额外计算负担的交换，SE-ResNet-50的准确性超过了ResNet-50的准确性，并且实际上接近了需要~7.58GFLOPs的更深的ResNet-101网络的准确性。

FC层的权重参数引入的总数由下式给出：

![](https://img-blog.csdnimg.cn/5680c30e158b42e2aafa2c534d676491.webp)

其中r表示缩减率，S表示阶段数（阶段是指在公共空间维度的特征图上操作的块的集合），Cs表示输出通道的维度，Ns表示阶段重复块的数量（当偏置项用于FC层时，引入的参数和计算成本通常可以忽略不计）。SE-ResNet-50引入了超过250万个额外参数。

1. 一个block是2C^2/r ；
2. 一个stage有N个block ；
3. 一个模型有S个stage 所以得到以上公式；

SEblock插入CNN中是很灵活的，因此有多个方式。
![](https://img-blog.csdnimg.cn/4d5f8f6122dd42b38d341b384fa404a3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_17,color_FFFFFF,t_70,g_se,x_16)

![](https://img-blog.csdnimg.cn/333e4cac1d5a4a02af378d0ca1a6bc34.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

三种变体：(1)SE-PRE块，其中SEblock在残差单元之前移动；(2)SE-POST块，其中SE单元在与恒等分支求和后移动（在ReLU之后）和(3)SE-Identity 块，其中SE单元放置在与残差单元平行的恒等连接上。这些变体如图5 所示，每个变体的性能在表14中报告。我们观察到SE-PRE、SE-Identity和提议的SE块的性能相似。

SE-ResNet完整架构✊

![](https://img-blog.csdnimg.cn/2412575adb884c4cb8b1cb03245a15cd.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

（左）ResNet-50。（中）SE-ResNet-50。（右）带有32×4d模板的SE-ResNeXt-50。括号内列出了残差积木的特定参数设置的形状和操作，而在外面显示了一个阶段中堆叠的积木数量。Byfcin后面的内括号表示一个SE模块中两个全连接层的输出维度。

## 实验

**讨论点**

1. **横向对比**
![](https://img-blog.csdnimg.cn/49a1e434888b42a0a6655e120fccc089.png)![](https://img-blog.csdnimg.cn/1fd9ea44a10b4bf696a33d3cb4dc92a0.webp)

数值越小越好。

2.  **调整Reduction比率**

Reduction比率空值这Dense layer 1 的neuron的数量

![](https://img-blog.csdnimg.cn/86196e1b5365481f9b4a9777b3aadece.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_18,color_FFFFFF,t_70,g_se,x_16)

论文推荐r = 16

3.   **GAP v.s. GMP**
![](https://img-blog.csdnimg.cn/043aa86d7c294f70943c589036d85096.webp)

结果显示用Avg Pooling更好

4.   **Excitation 阶段中不同的Activation函数对比**
![](https://img-blog.csdnimg.cn/d58675d28a134da9825058643e1faa87.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

5.   **SE Block的不同位置**
![](https://img-blog.csdnimg.cn/c0e6c3e706fe47c3aad9a0a161ed3b74.webp)

结果显示结果都差不多。

6.   **SE Block在ResNet的不同位置**
![](https://img-blog.csdnimg.cn/0a7333b5d93b415a97b955c6b2239ba5.webp)

把SE Block都安插进去效果最好！在深层安插比浅层效果好一点。

7.   **Squeeze的有无带来的影响**
![](https://img-blog.csdnimg.cn/3f5c04bfb9c94e6d8afec7dd968484e8.png)

肯定是有Squeeze效果好了。

8.   **对Excitation的探索**
![](https://img-blog.csdnimg.cn/53dde57bb9714154a9c5bb4226644e30.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

早期的Layer更加general，后期的Layer更Specific，5-2是一个拐点。

移除后期的layer可以减少param，同时模型不会受太大的影响。

## 总结
SENet对卷积层通道数进行权值评分，可以很好与其他网络(VGG, ResNet)结合。

相比于增加模型宽度(WRN中的width, ResNeXt中的cardinality)，深度(depth)，SE Block权重通道值，增加的参数少，增加的计算量小，增强效果好

**最后说一句，中秋快乐！**

![](https://img-blog.csdnimg.cn/fdf4dc6565fb4af8b7049d1656b465fd.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)









