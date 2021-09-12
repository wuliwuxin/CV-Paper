---
title: AlexNet
date: 2021-06-28 22:19:59
author: wulixin
tags: 论文
summary: AlexNet被认为是计算机视觉领域最有影响力的论文之一。
categories: 计算机视觉
---

# 带你读论文系列之计算机视觉--AlexNet
**少一些功利主义，多一些不为什么的坚持，你将变得异常美丽**！加油，小可爱们！
![](https://img-blog.csdnimg.cn/20210622225925769.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

## 背景及作者
**AlexNet**被认为是计算机视觉领域最有影响力的论文之一，它刺激了更多使用卷积神经网络和GPU来加速深度学习的论文的出现。截至2020年，AlexNet论文已被引用超过54,000次。

AlexNet参加了2012年9月30日举行的ImageNet大规模视觉识别挑战赛，达到最低的15.3%的Top-5错误率，比第二名低10.8个百分点。

**TOP5错误率**:即对一个图片，如果概率前五中包含正确答案，即认为正确。

**Hinton**是对比散度算法的发明人之一，也是反向传播算法和深度学习的积极推动者，被誉为“深度学习之父”。Hinton因在深度学习方面的贡献与Yoshua Bengio和Yann LeCun一同被授予了2018年的图灵奖。

![](https://img-blog.csdnimg.cn/20210622230043357.png)

**深度学习三驾马车：**
- 大规模数据；
- 硬件算法；
- 模型算法及调参；

**论文在如下背景下展开研究：**
- 当前图像分类任务主要是通过传统机器学习的方法进行的，模型容量小，且不易于实际使用，容易过拟合；
- 实际目标多样性丰富，标记好的数据集的样本数越来越大，需要更高容量的模型进行学习。而卷积神经网络可以通过调节深度和宽度来控制模型的容量，且充分利用了自然图像的局部空间相关性的特性；
- GPUs等硬件以及高度优化的2-D卷积运算的实现发展成熟，以足够强大，可用于训练较大的CNNs，结合如今的大数据集，不用过分担心过拟合；

## 论文

论文的模型，即AlexNet，其由多伦多大学，Geoff Hinton实验室设计，夺得了2012年ImageNet ILSVRC比赛的冠军，且其top-5错误率远低于第二名，分别为15.3%和26.2%。

AlexNet在**深度学习发展史上的历史意义远大于其模型的影响。**在此之前，深度学习已经沉寂了很久。在此之后，深度学习重新迎来春天，卷积神经网络也成为计算机视觉的核心算法模型。

**卷积过程中，卷积核的权重不变，局部连接，权值共享。**

**池化在卷积神经网络中的作用：**一方面可以减少feature map（或kennel）的尺寸，减小计算量；另一方面可以防止过拟合，把一些噪音过滤掉，同时为卷积神经网络引入了平移不**加粗样式**变性。


### 本文的主要内容有：

 - 使用 ReLU 激活函数加速收敛 
 - 使用 GPU 并行，加速训练。也为之后的分组卷积（group convolution）理论奠定基础。
 -  提出局部响应归一化（Local Response Normalization, LRN）增加泛化特性 (虽然被后人证明无效 )
- 使用交叠池化 (Overlapping Pooling) 防止过拟合 
- 提出Dropout，数据增强等手段防止过拟合


### 一、数据集：

数据来源于ImageNet，训练集包含120万张图片，验证集包含5万张图片，测试集包含15万张图片，这些图片分为了1000个类别，并且有多种不同的分辨率，但是AlexNet的输入要求是固定的分辨率，为了解决这个问题，Alex的团队采用低采样率把每张图片的分辨率降为256×256，具体方法就是给定一张矩形图像，首先重新缩放图像，使得较短边的长度为256，然后从结果图像的中心裁剪出256×256大小的图片。

**采用2块GPU对训练进行加速，仍需5-6天**。

网络包含许多新的和不寻常的特征，可以提高其性能并减少其训练时间。

网络的大小主要受当前GPU上可用内存量和我们愿意容忍的训练时间量的限制。

1000个类别中的每个类别中包含大约1000张图像总共有大约120万张训练图像、50,000张验证图像和150,000张测试图。

**举一个**🌰：训练集是平时作业题，验证集是模拟考试题，测试集是高考题，高考考砸说明过拟合了。

### 二、图像预处理：

每个图片变成256*256的图片，计算出每一个像素的均值，把每一个像素都减去对应的均值（相当于中心化的预处理）。

**预处理可以减少噪声的敏感性。**

### 三、激活函数

**为什么使用ReLu激活函数？**

**Sigmoid函数**：把（-∞,+∞）的任意数挤压在（0，1）小区间里；
**tanh函数**：把（-∞,+∞）的任意数挤压在（-1，1）小区间里；

sigmoid和tanh函数都是饱和激活函数，即当输入x过小或过大时，会被局限在一个很小的区域内，不能在进行变化。这种饱和的激活函数会造成梯度消失的问题，影响学习的效率和速度。

**ReLu函数为不饱和激活函数，可解决梯度消失问题。**

以ReLu函数代替Tanh或sigmoid函数（logistic回归使用的激活函数），这样能使网络训练，以更快的速度收敛。

在此之前，有人提出f(x)=|tanh(x)|，此方法在Caltech-101数据集上正则化和局部平均池化特别好。但是，在Caltech-101数据集上，**主要问题是防止过度拟合。**而对在**大型数据集上，更快的学习、训练对性能有很大影响**，此方法不满足。

![](https://img-blog.csdnimg.cn/20210622230545680.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

如图ReLU函数比tanh函数收敛快6倍。虽然不同网络结构效果可能不同，但ReLU普遍收敛很快。

### 四、GPU

**为什么要使用多GPU？**

单GPU无法训练太大的额数据集，因此使用两个并行GPU（当前GPU 能直接读取和写入彼此的内存）。

并行化本质上是将一半的内核（或神经元）放在一个GPU上，另一个GPU尽在某层中进行通信。

单GPU网络实际上与双GPU网络有相同数量的内核。这是因为网络的大部分参数都在第一个全连接层中，该层将最后的卷积层作为输入。

单GPU(半参数)模型中最后一个卷积和券链接层参数数量与双GPU(全参数)模型相同。**因此“半参数”并非真的只有一半的参数。**

**双GPU(全参数)的训练时间比但GPU(半参数)更短。**

### 五、局部响应归一化（LRN）

ReLU函数不像tanh和sigmoid一样有一个有限的值域区间，所以在ReLU之后需要进行归一化处理，LRN的思想来源于神经生物学中一个叫做“侧抑制”的概念，指的是被激活的神经元抑制周围的神经元。

Alex认为LRN可以防止过拟合，并与生物上的神经元类似；但是此方法在后续已不再采用了，如VGG指出有LRN和没有LRN没有区别，**LRN没有什么作用，只会徒劳的增加计算量。**

### 六、有重叠的池化

即池化步长小于池化窗口。

**Alex认为可以防止过拟合，但后续也不会采用这个技巧，为以后的模型进行了探索。**


### 七、整体结构
![](https://img-blog.csdnimg.cn/20210622230737565.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

224*224算出来第一层的feature map是一个小数，不符合整数的要求。则[(227-11)/4]+1=55，此处为作者的笔误。

第一个卷积层：96个11*11*3的卷积核步长为4；

第二个卷积层：256个5*5*48的卷积核；

第三个卷积层：384个3*3*256的卷积核；

第四个卷积层：384个3*3*192的卷积核；

第五个卷积层：256个3*3*192的卷积核；

全连接层：每层4096个神经元；

1000个神经元，它是线性分类，并没有非线性的激活函数，每一个神经元输出对应的分数，对1000个分数进行softmax，归一化，把它变成1000个概率，其和为1。

### 八、防止过拟合

存储1000个类别只需10bit，2的10次方=1024

但把图片映射成标签需要很多参数参数过多，可能会过出现严重的过拟合。

#### 1、增加训练样本

图像平移和水平翻转。将256*256的图像随机选取224*224的片段作为输入对象。一张图可以变成32*32*2=2048张图。虽然图像高度相似，但是可以有效的防止过拟合，扩充数据集。

先对图像做镜像反射，就像下图这样：

![](https://img-blog.csdnimg.cn/20210622230835216.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

摘自https://www.learnopencv.com/understanding-alexnet/

然后在原图和镜像反射的图（256×256）中随机抽取227×227的块，像这样：

![](https://img-blog.csdnimg.cn/20210622230857825.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

摘自https://www.learnopencv.com/understanding-alexnet/

#### 2、Dropout随机丢弃一定比例的神经元，被丢弃的神经元不参加训练过程。

Dropout做法是，对每一层、每次训练以概率P丢弃一些神经元，这样每次训练的网络都不一样。

**CNN的连接和参数要少得多，因此它们更容易训练。**

前两个全连接层中使用了dropout。在没有dropout 的情况下，网络严重的过拟合。dropout使收敛所需的迭代次数大约增加了一倍。

**训练结束后的测试流程，要用完成的网络结构，同时对该层的所有的参数（w,b）都乘以（1-P）。**

### 九、 细节、结果与讨论
![](https://img-blog.csdnimg.cn/20210622231015379.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

GPU1上的内核在很大程度上是不分颜色的，而GPU2上的内核在很大程度上是特定颜色的。这种特殊化发生在每次运行期间，并且与任何特定的随机权重初始化无关（除了GPU的重新编号之外）。

第一卷积层对224×224×3的输入图像学习了96个大小为11×11×3的卷积核。上面的48个核是在GPU1上学习的，下面的48个核是在GPU2上学习的。

初始化第二，第四和第五卷积层的神经元偏差，以及完全连接的隐藏层，恒定1。其余各层为数0。

所有层使用相同的的学习率。训练过程中，手动调整学习率，当验证错误率不再随当前的学习率提高时，将学习率除以10。初始值为0.01，在终止前减少了三次。

传统的无监督自动编码器没有使用性能，只提取原始像素空间特征，不提取语义特征。

另一种探测网络视觉知识的方法是考虑最后一个4096维隐藏层的图像所引起的特征激活。如果两幅图像产生的特征激活向量具有较小的欧氏分离度，我们可以说，神经网络的高层认为它们是相似的。

使用4096维实值向量之间的欧几里得距离来计算相似性是低效的，但是可以通过训练一个自动编码器将这些向量压缩成短的二进制代码来使其高效。

**如果去掉一个卷积层，网络的性能就会下降**。例如，去掉任何一个中间层，都会使网络的最高性能损失2%左右。（**ZFNet论文**中有更详细的去掉中间层性能的比较实验）

**没有使用无监督与训练。**

**此方法可用于视频序列中。**


> 愿你的每次流泪都是喜极而泣 
> 愿你精疲力竭有树可倚

