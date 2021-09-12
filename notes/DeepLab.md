---
title: DeepLab系列
categories: 计算机视觉
tags:
  - 语义分割
  - 论文
description: DeepLab系列一共有四篇文章，分别对应DeepLab V1、DeepLab V2、DeepLab V3和eepLab V3+。
date: 2021-08-08 20:50:39
---

#  语义分割模型之DeepLab系列
![空气新鲜，风景宜人](https://img-blog.csdnimg.cn/1fc6027f0a1d49748c36a8bc65ebe82b.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

## 前言
DeepLab系列一共有四篇文章，分别对应DeepLab V1、DeepLab V2、DeepLab V3和DeepLab V3+。

### DeepLab V1
论文题目：[Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)

开源代码：[TheLegendAli/DeepLab-Context](https://github.com/TheLegendAli/DeepLab-Context)

### DeepLab V2
论文题目：[DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)

开源代码：[DrSleep/tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet)

### DeepLab V3
论文题目：[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

开源代码：[leonndong/DeepLabV3-Tensorflow](https://github.com/eveningdong/DeepLabV3-Tensorflow)

### DeepLab V3+
论文题目：[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

开源代码：[jfzhang95/pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

### DeepLab系列的思想

图像分割CNN是根据classification这种high-level semantics改编的，但CNN做语义分割时精准度不够，根本原因是 DCNNs 的高级特征的平移不变性，即高层次特征映射，根源于重复的池化和下采样会丢失localization信息，即无法对像素点精确定位语义（low-level semantics）。

针对下采样或池化降低分辨率，DeepLab采用了空洞卷积代替池化操作来扩展感受野，获取更多的上下文信息。同时DeepLab v1v2 结合了深度卷积神经网络（DCNNs）和概率图模型（DenseCRFs）的方法。DeepLab v2提出了串行的ASPP模块，ASPP增强了网络在多尺度下多类别分割时的鲁棒性， 使用不同的采样比例与感受野提取输入特征，能在多个尺度上捕获目标与上下文信息，虽然大大扩展了卷积核的感受野，但**随着感受野越来越接近图像大小，会退化为1x1卷积。**

为了解决这个问题，**DeepLab v3改进了ASPP空洞卷积空间金字塔池化层，不同的dilation卷积并行操作，然后归一尺寸后求和**。ASPP模块借鉴PSPNet思想，通过不同采样率的空洞卷积并行采样，捕捉图像不同尺度的上下文信息。

DeepLab v3+通过添加一个简单而有效的解码器模块扩展DeepLab v3以优化分割结果，在PASCAL VOC 2012数据集和Cityscapes数据集中分别取得了89%和82.1%的MIOU。

### 语义分割面临的主要挑战

**分辨率**
连续的池化或下采样操作会导致图像的分辨率大幅度下降，从而损失了原始信息， 且在上采样过程中难以恢复。因此，越来越多的网络都在试图减少分辨率的损失， 比如使用空洞卷积，或者用步长为2的卷积操作代替池化。

**多尺度特征**
同一张图片中不同大小物体的分割精度不同，因为不同尺度卷积核对不同大小物体的分割效果不同。在分辨率较小的情况下，小物体的位置信息经常被丢失，通过设置不同参数的卷积层或池化层， 提取到不同尺度的特征图。将这些特征图送入网络做融合，对于整个网络性能的提升很大。但是由于图像金字塔的多尺度输入，造成计算时保存了大量的梯度，从而导致对硬件的要求很高。

## 1 DeepLab V1
DeepLab v1是结合了深度卷积神经网络（DCNNs）和概率图模型（DenseCRFs）的方法

**深度卷积神经网络（DCNNs）**

- 采用FCN思想，修改VGG16网络，得到 coarse score map并插值到原图像大小
- 使用Atrous convolution得到更dense且感受野不变的feature map

**概率图模型（DenseCRFs）**

- 借用fully connected CRF对从DCNNs得到的分割结果进行细节上的refine。

DeepLab v1:VGG16+空洞卷积+CRF对边缘分割结果进行后处理。针对下采样或池化降低分辨率，DeepLab采用了空洞卷积来扩展感受野，获取更多的上下文信息。同时，采用完全连接的条件随机场（CRF）提高模型捕获细节的能力。

感受域和步长以及卷积核之间的对应关系：

![](https://img-blog.csdnimg.cn/a8f41fd5c21e4c74b0d31e9364a34143.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
![](https://img-blog.csdnimg.cn/4e1d625a98ca4387a30e25cdca49970c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)![](https://img-blog.csdnimg.cn/b61c961872bc4924984567fae16a4a2c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

### DeepLab v1 的网络结构

1. 把全连接层（fc6、fc7、fc8）改成卷积层（端到端训练）
2. 把最后两个池化层（pool4、pool5）的步长2改成1（保证feature的分辨率下降到原图的1/8）。
3. 把最后三个卷积层（conv5_1、conv5_2、conv5_3）的dilate rate设置为2，且第一个全连接层的dilate rate设置为4（保持感受野）。
4. 把最后一个全连接层fc8的通道数从1000改为21（分类数为21）。
5. 第一个全连接层fc6， 通道数从4096变为1024， 卷积核大小从7x7变为3x3，后续实验中发现此处的dilate rate为12时（LargeFOV），效果最好。

![](https://img-blog.csdnimg.cn/4a8ad99eb87642d981a39d73b965d6ab.png)


### 实验设置

网络变形：

DeepLab-MSc：类似FCN，加入特征融合

DeepLab-7×7：替换全连接的卷积核大小为7× 7

DeepLab-4×4：替换全连接的卷积核大小为4× 4

DeepLab-LargeFOV：替换全连接的卷积核大小为3×3，空洞率为12。


![](https://img-blog.csdnimg.cn/e7d2acf16f7c43beb9201d210deaa11c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

损失函数：交叉熵 + softmax

优化器：SGD + momentum 0.9

batchsize：20

学习率：10^−3（每经过2000个epoch，学习率 * 0.1）

![](https://img-blog.csdnimg.cn/28f5116a5f3a49e9b4b0e2cf7dcc488e.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

## 2 DeepLab V2
![](https://img-blog.csdnimg.cn/79368d35d18c44baa83cf82934c2a179.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

Deeplab v2：VGG16/ResNet+串行的ASPP模块+CRF对边缘分割结果进行后处理。添加了ASPP空洞卷积空间金字塔池化层，通过不同的dilation卷积串行操作，来取代导致浅层特征损失的池化操作，大大扩大了感受野。

### 背景知识
#### 空洞卷积（Atrous Convolution）

**稠密映射：**

标准3×3卷积：3*3大小的区域对应一个输出值

空洞卷积(rate=2)：5*5大小的区域对应一个输出值
![](https://img-blog.csdnimg.cn/bf1eed852a744827b67d5c8149644d9f.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
![](https://img-blog.csdnimg.cn/b529ef0fafeb4bd08c2d93d4ad979a10.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

标准3×3卷积（rate为1），感受野为3；空洞卷积（rate为2），卷积核尺寸为5x5，感受野为7；空洞卷积（rate为4），卷积核尺寸为9x9，感受野为15。

![请添加图片描述](https://img-blog.csdnimg.cn/8a492b155ee043c3b1387185d64cecf6.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

#### ResNet

变化率：残差的引入去掉了主体部分，从而突出了微小的变化。

主要思想：用一个神经网络去拟合y=x这样的恒等映射，比用一个神经网络去拟合y=0这样的0映射要难。因为拟合y=0的时候，只需要将权重和偏置都逼近0就可以了。

![](https://img-blog.csdnimg.cn/340184514adc45a3921c12dbdd109c3c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
![](https://img-blog.csdnimg.cn/be07ad33cbb642d3bd7acf079bdf53bb.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

### Network&ASPP

ASPP模块构成——>DeepLab v1到DeepLab v2的进化——>基于VGG16的DeepLab v2在v1的基础上做了进一步调整（FC6-FC8替换为ASPP）

![](https://img-blog.csdnimg.cn/b842e05ebe80484cb3ea615aa30ba410.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/aab30393be6444d6894618db556e3d44.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

### 实验设置

损失函数：交叉熵 + softmax

优化器：SGD + momentum 0.9

Batchsize：20

学习率策略：step：10^−3（每经过2000个epoch，学习率 * 0.1）

poly：
![](https://img-blog.csdnimg.cn/22a9897753804089839eb9c42ca6d1e5.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

网络变形：

LargeFOV：3×3卷积 + rate=12(DeepLab v1最好结果)

ASPP-S：r = 2, 4, 8, 12

ASPP-L：r = 6, 12, 18, 24

![](https://img-blog.csdnimg.cn/892a2d8ff7de4c868abeefc1ee873650.png)![](https://img-blog.csdnimg.cn/4f8af814fad5448ba65d2d83cde09262.png)

## 3 DeepLab V3

![](https://img-blog.csdnimg.cn/db29db5edd9a49bc886c88d673509bee.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

Deeplab v3：ResNet+改进后的并行ASPP模块。随着采样率的增大，有效滤波器权重的数量（应用于有效特征的权重而不是padding补充的0）变少，在空洞率接近特征映射大小的极端情况下，3×3滤波器不是捕获整个图像上下文，而是退化为简单的1×1卷积（只有中心滤波器权重是有效的）。因此，v3使用了并行ASPP模块，最后一个分支拼接全局池化模块来捕获全局上下文信息。

**语义分割常用特征提取框架**
1. 图像金字塔：从输入图像入手，将不同尺度的图像分别送入网络进行特征提取，后期再融合。
2. 编解码结构：编码器部分利用下采样进行特征提取，解码器部分利用上采样还原特征图尺寸。
3. 深度网络vs空洞卷积：经典分类算法利用连续下采样提取特征，而空洞卷积是利用不同的采样率。
4. 空间金字塔结构：除ASPP外，仍有其他网络使用了该思想，如SPPNet、PSPNet等。

![](https://img-blog.csdnimg.cn/5c2621c7e62c433cb2d955fde3babed4.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

### 网络结构

经典分类算法网络架构，如ResNet  ——>  DeepLab v3空洞卷积串行网络结构  ——>  DeepLab v3空洞卷积并行网络结构（调整了ASPP模块）

![](https://img-blog.csdnimg.cn/8a8614ad97e441d3941bf2e14a69c68e.png)
![](https://img-blog.csdnimg.cn/b90c7a3e5d704250813cda1bfe7647bd.png)
![](https://img-blog.csdnimg.cn/6de7fa641b194aef9cb714c9e0545635.png)

            
### 实验设置

**裁剪尺寸**：裁剪图片至513x513（为了更好的拟合空洞率）

学习率策略：采用poly策略，原理同v2

**BN层策略**：当output_stride=16时，batchsize=16，同时BN层做参数衰减decay=0.9997。在增强的数据集上，以初始学习率0.007训练30K后，冻结BN层参数。当output_stride=8时，batchsize=8，使用初始学习率0.001训练30K。

## 4 DeepLab V3+
![](https://img-blog.csdnimg.cn/3fce4d09fb3b4e269eea88763b02c23d.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

Deeplab v3+：DeepLabv3+的核心是通过添加一个简单而有效的解码器模块来恢复对象边界（沿着对象边界来细化分割结果），扩展了DeepLab v3。以Xcepition/ResNet为骨架，采用深度可分离卷积进行编码，在多尺度特征提取ASPP模块后再接一个简单的解码器模块。

### 背景知识

#### 深度可分离卷积

**标准卷积：**
标准输入图片尺寸为12×12×3，用1个5×5×3的卷积核进行卷积操作，会得到8×8×1的输出；
用256个5×5×3的卷积核进行卷积操作，会得到8×8×256的输出。
参数计算：256×5×5×3 = 19200
![](https://img-blog.csdnimg.cn/c89edb27b5024cfdab086cce6c5fa16b.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
![](https://img-blog.csdnimg.cn/1688fe0d78bd44419c859dfb5dadcf65.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

**分组卷积：**
组卷积是对输入特征图进行分组，每组分别进行卷积。
假设输入特征图的尺寸为C*H*W (12× 5×5)，输出特征图的数量为N (6)个，如果设定要分成G (3)个groups，则每组的输入特征图数量为C/G (4)，每 组 的 输出特征图数量为N/G (2)，每个卷积核的尺寸为(C/G)*K*K (4×5×5)，卷积核的总数仍为N (6)个，每组的卷积核数量为N/G (2)，每个卷积核只与其同组的输入特征图进行卷积，卷积核的总参数量为N*(C/G)*K*K，可见，总参数量减少为原来的1/G。

深度可分离卷积是组卷积的一种极端情况，也就是输入有多少个通道，对应的分组就有多少个组，即分组的组数=输入特征图的通道数。
![](https://img-blog.csdnimg.cn/f3d42843851b418198593087b1e631f0.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
![](https://img-blog.csdnimg.cn/a5c19f7d3e2b45ae9b32b0ca92eddd5b.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

**深度可分离卷积 = 深度卷积 + 逐点卷积**

**深度卷积**：每个5×5×1的卷积核对应输入图像中的一个通道，得到三个8×8×1的输出， 拼接后得到8×8×3的结果

**逐点卷积**：设置256个1×1×3的卷积核，对深度卷积的输出再进行卷积操作，最终得到
8×8×256的输出

**参数计算：**
深度卷积参数 = 5×5×3 = 75

逐点卷积参数 = 256×1×1×3 = 768

总参数 = 75 + 768 = 843 << 19200

![](https://img-blog.csdnimg.cn/ab7f954841c742f0a3e8cfb3bed2deb9.png)![](https://img-blog.csdnimg.cn/88a7b178f787413b988845637267109a.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

### 网络结构

**编码器：**
1. 使用DeepLab v3作为编码器结构，输出与输入尺寸之比16(output_stride = 16)。
2. ASPP：一个1×1卷积 + 三个3×3卷积(rate = {6, 12, 18}) + 全局平均池化。

**解码器：**
1. 先把encoder的结果上采样4倍（双线性插值），然后与编码器中相对应尺寸的特征图进行拼接融合，再进行3x3的卷积， 最后上采样4倍得到最终结果
2. 融合低层次信息前，先进行1x1的卷积， 目的是降低通道数。

![](https://img-blog.csdnimg.cn/1da65cd0dc424bd6909775d7fb95481c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

DeepLab v3+对Xception进行了微调：

![](https://img-blog.csdnimg.cn/3c2a712dda4747999a01a4b8a821a00e.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
![](https://img-blog.csdnimg.cn/464b22aff4be4d0baab40e86d8807878.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
![](https://img-blog.csdnimg.cn/312255e342ef4d718f1c594f88c1518a.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

1. 更深的Xception结构，原始middle flow迭代8次，微调后迭代16次。
2. 所有max pooling结构被stride=2的深度可分离卷积替代。
3. 每个3x3的depthwise convolution（结合了空洞卷积）后都跟BN和Relu。

### 实验设置

剪尺寸：裁剪图片至513*513

学习率策略：采用poly策略，原理同v2 v3

![](https://img-blog.csdnimg.cn/8cdd6315658b41bdbfd6991e1d0bde7e.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)
![](https://img-blog.csdnimg.cn/d6b6db83db33440f83c2ee466866ed94.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

## 论文总结

**DeepLab系列发展历程**

v1：修改经典分类网络(VGG16)，将空洞卷积应用于模型中，试图解决分辨率过低及提取多尺度特征问题，用CRF做后处理（VGG16+空洞卷积+CRF对边缘分割结果进行后处理）

v2：设计ASPP模块，将空洞卷积的性能发挥到最大，沿用VGG16作为主网络，尝试使用ResNet-101进行对比实验，用CRF做后处理（VGG16/ResNet+串行的ASPP模块+CRF对边缘分割结果进行后处理）

v3：以ResNet为主网络，设计了一种串行和一种并行的DCNN网络，微调ASPP模块，取消CRF做后处理（ResNet+改进后的并行ASPP模块）

v3+：以ResNet或Xception为主网络，结合编解码结构设计了一种新的算法模型，以v3作为编码器结构，另行设计了解码器结构，取消CRF做后处理（ResNet/Xception+并行的ASPP模块+编码器结构）

![](https://img-blog.csdnimg.cn/0e4dda37ed6f41d4bb1245ef9fc9d198.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/23d1c632752747d0b5355e5ead2f5230.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1bGlfeGlu,size_16,color_FFFFFF,t_70)







