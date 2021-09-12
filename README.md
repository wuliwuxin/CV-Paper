简体中文 | [English](README_EN.md)
# CV-Paper

本项目主要是方便新手学习和自我回顾。经典之所以是经典，在于它独特的思想和质量的大幅提升。此项目主要是关于CV方向的论文，也可能会有NLP方向的部分文章。以下是我个人的一些经验，仅供参考。

持续更新中......

## 前言
### 必学内容--书籍篇
1. “西瓜书”周志华老师的[《机器学习》](./book)。公式推导配套使用[“南瓜书”](https://datawhalechina.github.io/pumpkin-book/#/)

2. [《统计学习方法》](./book)--李航老师。

3. " 花书"[《深度学习deep learning》](./book)--2018年图灵奖潜心打造的AI圣经。

4. [《动手学深度学习》](https://zh-v2.d2l.ai/)--亚马逊团队的李沐老师（沐神）深度学习可平易近人，教会你概念、背景和代码。

### 必学内容--视频篇

1. python推荐周沫凡的[莫烦python](https://mofanpy.com/)。Python基础 非常适合刚入门, 或者是以前使用过其语言的朋友们, 每一段视频都不会很长, 节节相连, 对于迅速掌握基础的使用方法很有帮助。一个默默奉献分享自己所学的东西的大佬。


2. 李宏毅的关于深度学习和机器学习的课程。李宏毅的视频讲得非常非常好，而且详细，十分惊叹竟然会有人 讲课讲得这么好。[李宏毅机器学习](https://www.bilibili.com/video/BV1Wv411h7kN?from=search&seid=12925803382584433478&spm_id_from=333.337.0.0)，[李宏毅深度学习](https://www.bilibili.com/video/BV1JE411g7XF?from=search&seid=12925803382584433478&spm_id_from=333.337.0.0)


3. 浙江大学胡浩基老师[研究生机器学习课程](https://www.bilibili.com/video/BV1dJ411B7gh)。人手写板书，而且讲的真的太好了，数学推导过程讲的一清二楚。可以搭配李航《统计学习方法》一同食用。很多书中没讲清楚的推导，听了胡老师的讲解后恍然大悟。一个讲课讲到全身湿透的老师。


4. 李飞飞的 [csn231 课程](https://www.bilibili.com/video/BV1nJ411z7fe?from=search&seid=6244444160908607298&spm_id_from=333.337.0.0)。这个课程我没有完整看过。但这是很多人都推荐必看的一个入门视频。


5. 吴恩达老师斯坦福[机器学习课程](https://www.bilibili.com/video/BV1VK4y1d7fj)。英语底子好的小伙伴可以看吴恩达大佬的全英视频。




* [图像分类](#图像分类)
* [目标检测](#目标检测)
* [图像分割](#图像分割)
* [医学影像相关](#医学影像相关)
* [CV中的Attention机制](#CV中的Attention机制)
* [其他](#其他)
* [常用数据集](#常用数据集)



## 图像分类
- [LeNet](https://ieeexplore.ieee.org/document/726791)

![](/picture/LeNet-5.png)

 1998年《Gradient-Based Learning Applied to Document Recognition》 CNN的开山之作，也是手写体识别经典论文。只要包含了卷积层的网络都可理解为卷积神经网络

- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

 2012年《ImageNet Classification with Deep Convolutional Neural Networks 》 ILSVRC-2012 大赛冠军，促进CNN的扛鼎之作，AlexNet是CNN发展史上的一个历史性转折，不能不读。
 
 [论文理解](/notes/AlexNet.md)

- [ZFNet](https://arxiv.org/abs/1311.2901)

AlexNet复杂模型的内部运作和行为，或它们如何取得如此好的性能，仍然没有什么深入了解。你如此优秀，我还不知道你为什么这么优秀。嗯，我必须要了解你一下。「🤔」

ZFNet 正是对 AlexNet 进行可视 化后改进而来，获得了 ILSVRC2014 的冠军。

[论文理解](/notes/ZFNet.md)

- [VGGNet](https://arxiv.org/abs/1409.1556)

 《Very Deep Convolutional Networks for Large-Scale Image Recognition》,虽然不是那年ImageNet大赛的冠军(那年的冠军是GoogLeNet),但是VGGNet对后面的ResNet，Inception产生了重要的影响。

 [论文理解](/notes/VGG.md)

- [GoogleNet](https://arxiv.org/abs/1409.4842)和[Inception V3](https://arxiv.org/abs/1512.00567)

 GoogleNet（Inception V1）和Inception V3:《Going Deeper with Convolutions》,《Rethinking the Inception Architecture for Computer Vision》,2014年ImageNet大赛冠军,Inception结构的设计很巧妙  。
 
  [论文理解](/notes/GoogleNet.md)
 
- [ResNet](https://arxiv.org/abs/1512.03385)

 《Deep Residual Learning for Image Recognition》，直接将top5错误率降到了3.57%（GoogLeNet 是6.66%），超越了人眼，文中最大的亮点就是残差块结构的设计。
 
  [论文理解](/notes/ResNet和ResNeXt.md)
 
- [ResNeXt](https://arxiv.org/abs/1611.05431)

 [论文理解](/notes/ResNet和ResNeXt.md)


* [Inception-v4, Inception ResNet](https://arxiv.org/abs/1602.07261)
    *  [Inception-v4代码实现](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py)
    *  [Inception-Resnet-V2代码实](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py)


- [MobileNet v1](https://arxiv.org/abs/1704.04861)


- [MobileNet v2](https://arxiv.org/abs/1704.04861)


- [MobileNet v3](https://arxiv.org/abs/1905.02244)


- [ShuffleNet v1](https://arxiv.org/abs/1707.01083)


- [ShuffleNet v2](https://arxiv.org/abs/1807.11164v1)


- [EfficientNet_v1](https://arxiv.org/abs/1905.11946v3)


- [EfficientNet_v2](https://arxiv.org/abs/2104.00298)


- [SENet](https://arxiv.org/abs/1709.01507)



## 目标检测
RCNN系列中包括RCNN, Fast RCNN, Faster RCNN, Mask RCNN

YOLO系列包括YOLO v1, YOLO v2, YOLO v3 以及SSD

- [RNN](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)


- [RCNN](https://arxiv.org/pdf/1311.2524v5.pdf)

  [论文理解](/notes/RCNN.md)



- [Fast-RCNN](https://arxiv.org/abs/1504.08083)


- [Faster-RCNN](https://arxiv.org/abs/1506.01497)

RCNN与SPPnet一些缺点与不足：

- **训练过程是一个multi-stage pipline**. RCNN首先在给定的region proposal上使用log损失进行微调。然后将卷积神经网络提取到的特征训练SVM分类器，利用SVM替代神经网络分类算法中常用的softmax。第三部分就是学习检测框的回归。
- **训练需要大量的空间与时间**， 由于训练过程中需要将卷积神经网络提取的特征写入磁盘，因此需要大量的物理存储空间，训练过程十分缓慢。
- **检测过程非常缓慢**，在测试时，从每个测试图像中的每个目标候选框提取特征。
RCNN主要对于每张图像的每个region proposal都输入CNN“网络进行计算，没有及逆行相应的共享计算，而SPPnet是利用共享卷积计算的方式来加速RCNN的检测过程，SPPnet将整张图片输入CNN网络得到特征图，然后利用空间金字塔池化网络对每个region proposal区域的特征图进行处理得到固定维度的特征向量，然后训练SVM分类器。

为解决上述优点， Fast RCNN主要贡献在于：

     1.Fast RCNN具有更高的目标检测的精度    
     2.训练过程采用多任务的损失函数
     3.训练可以更新所有网络层的参数
     4.不需要额外的磁盘空间存储特征


- [FPN](https://arxiv.org/abs/1612.03144)


- [SSD](https://arxiv.org/abs/1512.02325)


- [RetinaNet](https://arxiv.org/abs/1708.02002)

RetinaNet是继SSD和YOLO V2公布后，YOLO V3诞生前的一款目标检测模型，出自何恺明大神的《Focal Loss for Dense Object Detection》。全文针对现有单阶段法（one-stage)目标检测模型中前景(positive)和背景(negatives)类别的不平衡问题，提出了一种叫做Focal Loss的损失函数，用来降低大量easy negatives在标准交叉熵中所占权重（提高hard negatives所占权重)。为了检测提出的Focal Loss损失函数的有效性，所以作者就顺便提出了一种简单的模型RetinaNet。


- [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)


- [SPPNet](https://arxiv.org/abs/1406.4729)

github(PyTorch): https://github.com/yueruchen/sppnet-pytorch

用于视觉识别的深度卷积神经网络中的空间金字塔池。SPP-Net的提出一方面解决了固定的输入图像大小问题，即输入图像大小不受限制。因为SPP-Net解决了全连接层的输入需要固定维度的问题。

那么为什么全连接层的输入需要固定维度呢？

答：全连接层的计算其实相当于输入的特征图数据矩阵和全连接层权值矩阵进行内积。在配置一个网络时，全连接层的参数维度是固定的，所以两个矩阵要能够进行内积，则输入的特征图的数据矩阵维数也需要固定[X1][X2]。

RCNN的不足之处就是：
- 每一张图片会提取大约2千个候选区域（region Proposal），针对每个Region Proposal 都重复的使用 CNN提取特征，因此会在特征提取阶段耗费大量的时间
- 由于全连接层的输入维度是固定的，所以必须 resize（crop/wrap），Region Proposal 才能进行特征提取，结果会导致丢失图像信息和图片形变，影响特征提取的准确率。

所以SPPNet 针对R-CNN两处不足做了改进：
- 将Selective Search的Region Proposal不放入CNN进行特征提取，而是直接把原图片放入CNN进行特征提取，然后根据 Region Proposal位置在 conv5 的 feature map 做一个特征映射，再截取出每一个Region Proposal 所映射的 feature map。这样就避免了重复性用 CNN 对每个 Region Proposal 单独提取特征，减少了大量时间。
- SPPNet 在原来的CNN的conv5之后加入了 Spatial  Pyramid Pooling layer（空间金字塔池化层）替换掉原来的 Pooling5 layer，由于SPP layer 可以接受不同 size 的feature maps 并输出相同 size 的feature maps，因此避免了 resize而导致的图片形变问题。
　　总结一下，SPP-net的初衷非常明晰，就是希望网络对输入的尺寸更加灵活，分析到卷积网络对尺寸并没有要求，固定尺寸的要求完全来源于全连接层部分，因而借助空间金字塔池化的方法来衔接两者，SPP-net 在检测领域的重要贡献是避免了R-CNN的变形，重复计算等问题，在效果不衰减的情况下，大幅提高了识别速度。


## 图像分割
- [FCN](https://arxiv.org/abs/1411.4038)

（CVPR 2015 best paper honorable mention）如果你会分类网络，那么分割网络你也就会很快明白了，因为分类网络是把一张图预测成一类，而分割网络是把一张图每个像素都预测一下，其实是一样的。

- [R-FCN](https://arxiv.org/abs/1605.06409)

- DeepLab系列：[DeepLab V1](https://arxiv.org/abs/1412.7062v4)、[DeepLab V2](https://arxiv.org/abs/1606.00915)、[DeepLab V3](https://arxiv.org/abs/1706.05587)和[DeepLab V3+](https://arxiv.org/abs/1802.02611)，

 [论文理解](/notes/DeepLab.md)

## 医学影像相关
[DeepLung](https://arxiv.org/abs/1801.09555)

 工程地址：https://github.com/wentaozhu/DeepLung、https://github.com/uci-cbcl/DeepLung

 提出一个全自动的肺部CT癌症诊断系统DeepLung。DeepLung包括两部分，结节检测(识别候选结节位置)和分类（将候选结节分类为良性或恶性）。考虑到肺部CT数据的3D特性和双路径网络（DPN)的压缩性，设计了两个深度3D DPN分别用于结节检测和回归。特别地，一个带3D双路径块和U-net型编码-解码结构的Faster RCNN来高效的学习结节特征。对于结节分类，提出一个带3D双路径网络特征的梯度提升机（GBM)。在LIDC-IDRI的公开数据集络验证了结节分类子网，取得了比state-of-the-art更好的性能，并且在基于图像模式上超过了有经验医生。在DeepLung系统中，首先通过结节检测子网络检测出候选结节，然后使用分类子网络做结节诊断。广泛的实验结果表明，DeepLung在LIDC-IDRI数据集上的结节级别和患者级别诊断方面的性能均与经验丰富的医生相当。


## CV中的Attention机制
- [Transformer](https://arxiv.org/abs/1706.03762)

Transformer由论文《Attention is All You Need》提出，现在是谷歌云TPU推荐的参考模型。

**Motivation**:
1. 靠attention机制，不使用rnn和cnn，并行度高
2. 通过attention，抓长距离依赖关系比RNN强

**创新点**：

1. 通过self-attention，自己和自己做attention，使得每个词都有全局的语义信息（长依赖

2. 由于 Self-Attention 是每个词和所有词都要计算 Attention，所以不管他们中间有多长距离，最大的路径长度也都只是 1。可以捕获长距离依赖关系。

3. 提出multi-head attention，可以看成attention的ensemble版本，不同head学习不同的子空间语义。

- [Vision Transformer(ViT)](https://openreview.net/pdf?id=YicbFdNTTy)

1. 作者尝试将Transformer结构直接应用到图像上，即将一张图像分割成多个patches，这些patches看作是NLP的tokens (words)，然后对每个patches做一系列linear embedding操作之后作为Transformer的input。

2. 但作者发现，该方法在ImageNet这种 mid-sized  数据集上进行训练，acc要比同等大小的ResNets低一点。原因可能是，Transformers缺少CNN固有的inductive biases，比如说平移不变性和局部性，因此在训练集不够充足的时候，泛化性不好。
 
3. 但作者发现当在更大规模的数据集上pretrain后，再transfer 到其他任务上，Transformer却可以表现得更好。

 [论文理解](/notes/ViT.md)
 
- [Swin Transformer](https://arxiv.org/abs/2103.14030)

代码：[语义分割](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)，[图像分类](https://github.com/microsoft/Swin-Transformer)，[目标检测](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)

- [BoTNet](https://arxiv.org/abs/2101.11605)

 该架构将自注意力纳入了多种计算机视觉任务，包括图像分类，目标检测和实例分割。通过仅在ResNet的最后三个bottleneck blocks中用全局自注意力替换空间卷积，并且不进行其他任何更改，我们的方法在实例分割和目标检测方面显著改善了基线，同时还减少了参数，从而使延迟最小化。
 
 [论文理解](/notes/BoTNet.md)

## 其他

- [DeepID2+](https://arxiv.org/abs/1412.1265)

 《Deeply learned face representations are sparse, selective, and robust》为什么要推荐这篇论文呢？人脸识别领域，DeepID大名如雷贯耳，与DeepID,DeepID2不同的是，这篇论文并不是单纯讲人脸识别，论文深入分析了CNN的内部结构，试图从理论上解释CNN强大的特征提取能和分类识别能力，这是学者第一次试图去探索CNN的本质属性，看完这篇论文，相信对CNN会有更深入的了解。

- [GPT-3](https://arxiv.org/abs/2005.14165)

 github链接：https://github.com/openai/gpt-3

 对于所有任务，应用GPT-3无需进行任何梯度更新或微调，而仅通过与模型的文本交互指定任务和少量演示即可。 GPT-3在许多NLP数据集上均具有出色的性能，包括翻译，问题解答和完形填空任务，以及一些需要即时推理或领域适应的任务



## 常用数据集

* [ImageNet](#ImageNett)
* [COCO](#COCO)
* [CIFAR-10](#CIFAR-10)
* [PASCAL VOC](#PASCAL-VOC)

### [ImageNet](https://paperswithcode.com/dataset/imagenet)
 
 ImageNet数据集包含14,197,122张根据WordNet层次结构标注的图像。自2010年以来，该数据集被用于ImageNet大规模视觉识别挑战赛（ILSVRC），这是一个图像分类和物体检测的基准。公开发布的数据集包含一组人工标注的训练图像。一组测试图像也被发布，但不包括人工注释。ILSVRC的注释分为两类之一。(1)图像级别的注释，即对图像中是否存在一个物体类别的二进制标签，例如，"这幅图像中有汽车"，但 "没有老虎"，以及(2)物体级别的注释，即对图像中一个物体实例的紧密边界框和类别标签，例如，"有一把螺丝刀，中心位置是(20,25)，宽度为50像素，高度为30像素"。ImageNet项目不拥有图像的版权，因此只提供图像的缩略图和URL。
 
 - 非空的WordNet同义词总数：21841个
 - 图像总数。14197122
 - 有边界盒注释的图像数量。1,034,908
 - 具有SIFT特征的同义词组数量。1000
 - 具有SIFT特征的图像数量。1.2百万

### [COCO](https://paperswithcode.com/dataset/coco)
 MS COCO（Microsoft Common Objects in Context）数据集是一个大规模的物体检测、分割、关键点检测和字幕数据集。该数据集由328K图像组成。

 分割。MS COCO数据集的第一个版本是在2014年发布的。它包含16.4万张图像，分为训练（83K）、验证（41K）和测试（41K）集。2015年，又发布了81K的测试集，包括所有以前的测试图像和40K的新图像。

 根据社区的反馈，在2017年，训练/验证的分割从83K/41K改为118K/5K。新的分割使用相同的图像和注释。2017年的测试集是2015年测试集的41K图像的一个子集。此外，2017年的版本包含一个新的未注释的数据集，即123000张图像。

注释。该数据集有以下注解

- 物体检测：边界框和每个实例的分割掩码，有80个物体类别。
- 说明：图像的自然语言描述（见MS COCO说明）。
- 关键点检测：包含超过200,000张图片和250,000个标有关键点的人物实例（17个可能的关键点，如左眼、鼻子、右臀、右脚踝）。
- 东西图像分割--每像素分割掩码有91个东西类别，如草、墙、天空（见MS COCO Stuff）。
- 全景：全场景分割，有80个事物类别（如人、自行车、大象）和91个东西类别的子集（草、天空、道路）。
- 密集姿态：超过39,000张图片和56,000个人物实例被标注了密集姿态注释--每个被标注的人物都被标注了一个实例ID，以及属于该人物身体的图片像素和一个模板3D模型之间的映射。这些注释只对训练和验证图像公开。

### CIFAR-10

 CIFAR-10数据集（加拿大高级研究所，10类）是Tiny Images数据集的一个子集，由60000张32x32的彩色图像组成。这些图像被标记为10个相互排斥的类别之一：飞机、汽车（但不是卡车或皮卡）、鸟、猫、鹿、狗、青蛙、马、船和卡车（但不是皮卡）。每类有6000张图像，每类有5000张训练图像和1000张测试图像。

 决定一个图像是否属于一个类别的标准如下。

- 类的名称应该在 "这张图里有什么？"这个问题的可能答案列表中名列前茅。
- 图片应该是照片般真实的。贴标签者被要求拒绝线描画。
- 图片应该只包含该类别所指的物体的一个突出实例。该物体可以是部分遮挡的，或者是从一个不寻常的角度看到的，只要贴标者仍然清楚其身份。


### PASCAL-VOC

PASCAL VOC数据集是目标检测领域比较知名的数据集，该数据集分为VOC2007和VOC2012两个子集，其官方下载地址如下：
* [VOC2007-trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
* [VOC2007-test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
* [VOC2012-trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
官网时常不稳定，进而导致下载较慢甚至下载失败，推荐一定稳定的镜像源地址：[YOLOv3作者主页](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)


## 赞助和支持

如果你觉得它对你很有帮助, 请你也分享给需要学习的朋友们。

如果你看好我的经验分享, 也请考虑适当的赞助打赏, 让我能继续分享更好的内容给大家。



<table width="100%" border="0" cellspacing="15" cellpadding="0">
<tbody>
    <tr>
          <th>支付宝</th>
          <th>微信</th>
      </tr>
  <tr>
    <td>
      <img height="150" src="https://github.com/wuliwuxin/CV-Paper/blob/main/picture/AliPay.jpg"/>
    </td>
    <td width="50%">
        <p align="left"> 
        <p align="left"> <img src="https://github.com/wuliwuxin/CV-Paper/blob/main/picture/wechatPay.jpg" /> </p>
    </td>
   </tr>
</tbody>
</table>
