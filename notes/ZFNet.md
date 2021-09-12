---
title: ZFNet
date: 2021-06-28 22:19:59
author: wulixin
tags: 论文
summary: AlexNet提出了一种可视化方法，展示了如何使用这些可视化来调试模型的问题以获得更好的结果。
categories: 计算机视觉
---
# 带你读论文系列之计算机视觉--FZNet
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625220117522.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
## 回顾
[《带你读论文系列之计算机视觉--AlexNet》](https://blog.csdn.net/wuli_xin/article/details/118121084?spm=1001.2014.3001.5501)

Convolutional Network表现好：
- 更大的训练集的可用性，有数百万的标记样本；
- 强大的GPU实现，使训练非常大的模型成为现实；
- 更好的模型正则化策略，如Dropout

复杂模型的内部运作和行为，或它们如何取得如此好的性能，仍然没有什么深入了解。你如此优秀，我还不知道你为什么这么优秀。嗯，我必须要了解你一下。「🤔」

## 背景与作者
**ZFNet**是**Matthew D.Zeiler与Rob Fergus**于2013年提出，并获得了**2013年ImageNet的冠军**。2012年AlexNet问世，并在ImageNet竞赛中取得了优异的成绩，也证明了大的卷积网路的性能优异，但是我们并不知道为什么CNN性能好。

因此，该论文是在AlexNet基础上进行了一些细节的改动,通过使用可视化技术揭示了神经网络各层到底在干什么，起到了什么作用。也是基于这个技术，作者对AlexNet进行了优化，**调整之后的网络的性能在很多问题上性能都好于AlexNet。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625220749108.png)
**Matthew D.Zeiler**创始人兼 CEO Matthew Zeiler 是机器学习博士。应用人工智能 (AI) 领域的先驱和思想领袖。Matt 与著名机器学习专家 Geoff Hinton 和 Yann LeCun 在计算机视觉方面的开创性研究推动了图像识别行业从理论到实际应用。自 2013 年创立 Clarifai 以来，Matt 已将他屡获殊荣的研究发展为开发人员友好的产品，使企业能够快速无缝地将 AI 集成到他们的工作流程和客户体验中。今天，Clarifai 是领先的独立 AI 公司，“被广泛视为机器学习领域最有前途的 [初创公司] 之一。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625220815254.png)
**Rob Fergus**的研究领域是计算机视觉、机器学习和计算机图形。他对建立图像的统计模型感兴趣，这些模型既包括高层次的物体和场景，也包括低层次的像素和边缘。这些模型可以在各种问题中部署。他特别感兴趣的问题包括：物体识别、图像搜索等。

## 论文
**论文的主要贡献：**
1、特征可视化
2、对于CNN结构的改进
3、对于遮挡的敏感性
4、关联分析
5、特征提取的通用性
6、特征分析

### 技术：
1、**反池化过程**：池化是不可逆的过程，然而我们可以通过记录池化过程中，最大激活值得坐标位置。然后在反池化的时候，只把池化过程中最大激活值所在的位置坐标的值激活，其它的值置为0。这个过程只是一种近似，因为我们在池化的过程中，除了最大值所在的位置，其它的值也是不为0的。
2、**反激活**：在Alexnet中，relu函数是用于保证每层输出的激活值都是正数，因此对于反向过程，我们同样需要保证每层的特征图为正值，也就是说这个反激活过程和激活过程没有什么差别，都是直接采用relu函数。
3、**反卷积**：利用相同卷积核的转置作为核，于输入做卷积运算

### 特征可视化

**什么是平移不变性？**
不变性即使目标的外观发生了某种变化，但是你依然可以把它识别出来。这对图像分类来说是一种很好的特性，因为我们希望图像中目标无论是被平移，被旋转，还是被缩放，甚至是不同的光照条件、视角，都可以被成功地识别出来。

所以上面的描述就对应着各种不变性：
- 平移不变性：Translation Invariance
- 旋转/视角不变性：
- Ratation/Viewpoint Invariance
- 尺度不变性：Size Invariance
- 光照不变性：Illumination Invariance

**特征可视化局限在第一层**，可以投射到像素空间，但是更**高层的解释方法有限**。因为对于较高的层，不变性是非常复杂的，所以很难被简单的二次逼近法所捕获。

**作者的方法**提供了一个非参数化的不变性观点，显示了训练集的哪些patch激活了特征图。

在2013年Donahue等人展示了在数据集中识别patch的可视化，这些patch负责模型中更高层次的强激活。我们的可视化不同的是，不只是输入图像的作物，而是自上而下的投影，揭示了每个patch内刺激特定特征图的结构。

带标签的图像来训练模型，交叉熵损失函数用来比较预测值与实际值。（卷积层中的过滤器、全连接层中的权重矩阵和偏置）通过反向传播损失对整个网络参数的导数进行训练，并通过随机梯度下降更新参数。

我们提出了一种将这些活动映射回输入像素空间的新方法，显示什么输入模式最初导致了特征图中的给定激活。

### 用反卷积法进行可视化

反卷积网络，相反的convolution network过程，而不是将像素映射到特征。convnet使用学习的过滤器来卷积来自前一层的特征图。deconvnet使用相同过滤器的转置版本（即垂直和水平翻转每个过滤器），但应用于校正后的map，而不是下层的输出。

**最大池化不可逆**，但我们可以在变量中记录每个池化区域内最大值的位置来近似逆转。反池化：在卷积神经网络时，记录每个最大池化局部最大的位置。反池化时，将最大位置值还原，其余位置设为0。

为了确保特征图始终是正的，我们通过ReLu非线性函数重建信号。单个激活获得的重建类似于原始输入图像的一小部分。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625221457777.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)


### 训练细节

一个区别是在CNN的第3 、4、5层中使用的稀疏连接（由于模型被拆分为2个GPU）在我们的模型中被密集连接替换。

每个RGB图像都通过将最小尺寸调整为256、裁剪中心256x256区域、减去每像素平均值（跨所有图像）然后使用10个不同的大小为224x224 的子裁剪（角+中心和（外）水平翻转）进行预处理。为每个训练示例生成多种不同的裁剪和翻转，以提高训练集的大小。使用小批量大小为128的随机梯度下降来更新参数。

在模型的第一层，输入图像大致在[-128,128]范围内。


**参数设置：**
- 初始学习率：0.01
- 初始动量：0.9
- dropout概率：0.5
- 所有初始权重:0.01
- 偏置项：0

将RMS值超过固定半径0.1的卷积层中的每个滤波器重新归一化为这个固定半径。

> 均方根误差（RMSE，root-mean-square error）：
均方根误差为了说明样本的离散程度。
做非线性拟合时,RMSE越小越好。

**该模型在70个epoch后停止训练，这在单个GTX580 GPU上花费了大约12天。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625221537984.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625221551891.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625221545124.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
**第2层**响应角和其他边缘/颜色连接；
**第3层**具有更复杂的不变性，捕获相似的纹理；
**第4层**显示出显着的变化，但更具有特定类别：
**第5层**显示了具有显着姿势变化的整个对象；

**注意：**
- 每个特征图内的强分组；
- 较高层次的不变性；
- 图像中歧视性部分的夸大，例如狗的眼睛和鼻子
### 训练时特征演变
在一个给定的特征图中投射回像素空间的强激活（跨越所有训练实例）的进展。外观上的突然跳动是由于最强激活的图像发生了变化。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625221733179.png)
上图表示通过训练随机选择的模型特征子集的演化。每个图层的特征显示在不同的块中。在每个块中，我们在 epoch [1,2,5,10,20,30,40,64] 中显示了随机选择的特征子集。可视化显示了给定特征图的最强激活（在所有训练示例中），使用我们的 deconvnet 方法投影到像素空间。人为增强色彩对比度。

**上层只有在相当多的 epochs(40-50)之后才发展，需要让模型训练直到完全收敛。**

### 特征不变性
样本图像，在观察模型顶层和底层的特征向量相对于未变换特征的变化时，它们被不同程度地平移、旋转和缩放。小的变换在模型的第一层有很大的影响，但在顶部特征层的影响较小。该网络的输出对平移和缩放是稳定的。一般来说，除了具有旋转对称性的物体，输出对旋转是不变的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625221819624.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
上图表示：模型内垂直平移、缩放和旋转不变性的分析（分别为 a-c 行）。第 1 列：5 个经过转换的示例图像。列2和3：分别来自第1层和第7层的原始图像和变换图像的特征向量之间的欧几里德距离。第4列：每个图像的真实标签的概率，因为图像被变换。

### 结构的选择

可视化 Krizhevsky 等人的架构**存在的问题**：
第一层滤波器是极高频和低频信息的混合，几乎没有覆盖中频。
第二层可视化显示了由第一层卷积中使用的大步幅 4 引起的混叠伪影。

**解决方案：**
将第 1 层过滤器的大小从 11x11 减小到 7x7 并且使卷积的步幅4变为 2。这样在第 1 层和第 2 层特征中保留了更多信息，它还提高了分类性能。

### 遮挡敏感度
对于**图像分类**方法，**存在的问题**是模型是真正识别图像中对象的位置，还是仅使用周围的上下文。

通过系统地用灰色方块遮挡输入图像的不同部分，并监测分类器的输出。因为当对象被遮挡时，正确类别的概率显着下降。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625221941959.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
如图所示三个测试示例，我们系统地用灰色方块（第 1 列）覆盖场景的不同部分，并查看顶部（第 5 层）特征映射（第二列和第三列）和分类器输出（第四列和第五列) 变化。第二列：对于灰度的每个位置，我们在第 5 层特征图（未遮挡图像中响应最强的那个）中记录总激活。第三列：投影到输入图像（黑色方块）中的该特征图的可视化，以及来自其他图像的该地图的可视化。第一行的例子显示了最强的特征是狗的脸。当这被掩盖时，特征图中的活动减少（第二列中的蓝色区域）。第四列：作为灰色方块位置的函数的正确类别概率图。例如。当狗的脸被遮挡时，“博美犬”的概率显着下降。第五列：最可能的标签作为遮挡器位置的函数。例如。在第一行，对于大多数位置，它是“博美犬”，但如果狗的脸被遮挡而不是球，那么它预测“网球”。在第二个示例中，汽车上的文本是第 5 层中最强的特征，但分类器对车轮最敏感。第三个示例包含多个对象。第 5 层中最强的特征挑出人脸，但分类器对狗敏感（第四列中的蓝色区域），因为它使用了多个特征图。

当遮挡物覆盖出现在可视化中的图像区域时，我们看到特征图中的活动急剧下降。**这表明可视化真正对应于对象的位置和该特征图的图像结构。**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625222041365.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625222047265.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
如图所示：不同狗图像中不同对象部分的对应性度量。眼睛和鼻子的较低分数（与随机对象部分相比）表明该模型在模型的第 5 层隐式建立了某种形式的部分对应关系。在第 7 层，分数更相似，可能是由于上层试图区分不同品种的狗。

### 实验

**ImageNet数据集**：训练集1.3M；验证集50K;测试集100K；1000个分类.

#### 网络架构
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625222128677.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
224 x 224（具有 3 个颜色平面）图像作为输入呈现。第一层96 个过滤器（红色）卷积的，每个过滤器的大小为 7 x 7，在 x 和 y 中使用 2 的步长。然后得到的特征图是：（i）通过一个修正的线性函数；(ii) 池化（最大在 3x3 区域内，使用步幅 2）；（iii）跨特征映射归一化的对比度，以提供 96 个不同的 55 x 55 元素特征映射。

在第 2、3、4、5 层重复类似的操作。最后两层是全连接的，将顶部卷积层的特征作为向量形式（6·6·256 = 9216 维）的输入。最后一层是一个C-way softmax function，C是类的数量。所有过滤器和特征图都是方形的。

架构变化（第1层的7×7过滤器和第1、2层的stride 2卷积），比他们的单一模型结果Top-5 error 降低了1.7%，证明了这样改进的正向效果。

结合多个模型时，我们得到14.8%的测试误差。

### 模型探索
- 删除模型的两个卷积层或者两个全连接层，对分类结果影响不大。
- 去除中间卷积层和全连接层后，只有4个层的模型的性能明显下降，说明了模型的整体深度很重要。
- 单独改变全连接层的尺寸，对分类结果影响不大，但增大中间卷积层的尺寸对分类结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625222219818.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
使用ImageNet预训练的模型用到其他数据集上也取得了很好的效果，说明了fine-tuning的价值。

特征分析部分在原始模型某一中间层后加SoftMax或者SVM分类器，说明了最后面层特征的分类效果最好，再次佐证了不同层之间学习到的特征具有层次结构，层数越大，学习到的特征表达能力越强。

**其他数据集上的表现**

对于 Caltech-101 和 Caltech-256，数据集非常相似，以至于我们模型得到结果做好。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625222515559.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625222523409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
但我们的 convnet 模型对 PASCAL 数据的泛化不太好，可能存在数据集偏差。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625222600421.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
### 总结
- 提出了一种可视化方法，展示了如何使用这些可视化来调试模型的问题以获得更好的结果。
- 发现学习到的特征远不是无法解释的，而是特征间存在层次性，层数越深，特征不变性越强，类别的判别能力越强；
- 通过可视化模型中间层，在AlexNet基础上进一步提升了分类效果；
- 通过一系列遮挡实验证明，该模型在进行分类训练时，对图像中的局部结构高度敏感，而不仅仅是使用广泛的场景上下文。对该模型的消融研究表明，网络的最小深度，而不是任何单个部分，对模型的性能至关重要。遮挡实验表明分类时模型和局部块的特征高度相关；
- 模型的深度很关键；
- 预训练模型可以在其他数据集上fine-tuning得到很好的结果。


> 愿你的身后总有力量
愿你成为自己的太阳





