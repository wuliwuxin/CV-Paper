---
title: RCNN
categories: 计算机视觉
tags:
  - 论文
description: RCNN系列的文章主要是RCNN，Fast RCNN， Faster RCNN， Mask RCNN, Cascade RCNN,这一系列的文章是目标检测two-stage算法的代表，这系列的算法精度高，效果好，是一类重要的方法。
date: 2021-09-01 08:33:47
---

# 带你读论文系列之计算机视觉RCNN
![但是这样，我们总是习惯了不愿改变。](https://img-blog.csdnimg.cn/c6dbf5873193455aa98f2dd163d645f6.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

## 前言

RCNN系列的文章主要是**RCNN，Fast RCNN， Faster RCNN， Mask RCNN, Cascade RCNN**,这一系列的文章是目标检测two-stage算法的代表，这系列的算法精度高，效果好，是一类重要的方法。
 
先来讲解目标检测开山之作R-CNN
 
[**Rich feature hierarchies for accurate object detection and semantic segmentation**](https://arxiv.org/abs/1311.2524)

这是一篇比较早的Object Detection算法，发表在2014年的CVPR，也是R-CNN系列算法的开山之作，网上可以搜到很多相关的博客讲解，本篇没有按论文顺序来讲述，而是结合自己经验来看这个算法，希望给初学者一个直观的感受，细节方面不需要太纠结，因为很多部分在后来的算法中都改进了。

R-CNN作为第一个将RPN理念与CNN结合的论文，在后续不断改进，诞生了Faster-RCNN，Mask-RCNN等一系列经典模型，所以是一篇入门CV必读的经典论文。

##  01 寻找推荐区域

在R-CNN中架构的第一步是要寻找推荐区域（Region Proposal），也就是找出可能的感兴趣区域（Region Of Interest, ROI）。获取推荐区域的方法有下三种，分为：滑动窗口、规则块和选择性搜索。
 
第一种就是滑动窗口。滑动窗口本质上就是穷举法，利用不同的尺度和长宽比把所有可能的大大小小的块都穷举出来，然后送去识别，识别出来概率大的就留下来。很明显，这样的方法复杂度太高，产生了很多的冗余候选区域，在现实当中不可行。
 
第二种是规则块。在穷举法的基础上进行了一些剪枝，只选用固定的大小和长宽比。但是对于普通的目标检测来说，规则块依然需要访问很多的位置，复杂度高。
 
第三种是**选择性搜索**。从机器学习的角度来说，前面的方法召回是不错了，但是精度差强人意，所以问题的核心在于如何有效地去除冗余候选区域。**其实冗余候选区域大多是发生了重叠，选择性搜索利用这一点，自底向上合并相邻的重叠区域，从而减少冗余。**

## 02 R-CNN结构

R-CNN主要由**三部分**结构构成，分别为：

**第一阶段**：Region Proposals提取阶段。主要使用的是之前就已经提出的技术：selective search，该技术我们不细讲，如感兴趣可在最后看一下该论文的具体实现细节。我们只要明白该阶段输入了图像，会返回大约2k个region proposals.

**第二阶段**：是在得到了很多region proposal后我们将其分别输入到一个CNN网络中，每一个region proposals输入一次，返回一个该region proposal对应的特征向量（4096维），在CNN的具体结构上，论文作者使用了Alexnet作为模型骨架。

有的人可能已经发现问题了，第一阶段提取的region proposal都是不同大小的，而CNN的输入往往都是要求固定的，在本文要求CNN的输入大小固定为227*227，那么我们如何解决region proposal的大小不匹配问题呢。

**第三阶段**：在得到了每个region proposal的特征向量后，我们使用SVM二分类器对每一个region proposal进行预测。

有同学又有疑问了，为啥这么麻烦，直接跟Alexnet一样用一个Softmax直接做分类不香么，干嘛还用SVM一个个去分类呢?

主要由于：正负样本类别不平衡，负样本的数量远大于正样本数量，训练过程中以128的batch_size为例，仅有32个正样本，96个负样本，而且负样本都是随机取样，而SVM有着比Softmax更好的利用困难样本的能力。
 
**正样本**

本类的真值标定框。

**负样本**

考察每一个候选框，如果和本类所有标定框的重叠都小于0.3，认定其为负样本。

## 03 R-CNN详情

R-CNN的工作流程

![](https://img-blog.csdnimg.cn/77895e7adc1741c1871ff45f3c6ecbce.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

RCNN算法分为4个步骤

- 一张图像生成1K~2K个候选区域；
- 对每个候选区域，使用深度网络提取特征；
- 特征送入每一类的SVM 分类器，判别是否属于该类；
- 使用回归器精细修正候选框位置；

VOC 2007 训练的扭曲训练样本。

![](https://img-blog.csdnimg.cn/a811c0b95737445e8b0d9b23abcca283.webp?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

![](https://img-blog.csdnimg.cn/7e51eab8401043cda8f85fc06d22bd2f.webp)

VOC 2010 测试的检测平均精度 (%)。R-CNN 与 UVA 和 Regionlets 最直接可比，因为所有方法都使用选择性搜索区域建议。边界框回归（BB）。在发布时，SegDPM 是 PASCAL VOC 排行榜上的最佳表现。DPM 和 SegDPM 使用其他方法未使用的上下文重新评分。

## 结果

论文发表的2014年，DPM已经进入瓶颈期，即使使用复杂的特征和结构得到的提升也十分有限。本文将深度学习引入检测领域，一举将PASCAL VOC上的检测率从35.1%提升到53.7%。

本文的前两个步骤（候选区域提取+特征提取）与待检测类别无关，可以在不同类之间共用。这两步在GPU上约需13秒。

同时检测多类时，需要倍增的只有后两步骤（判别+精修），都是简单的线性运算，速度很快。这两步对于100K类别只需10秒。

参考：
https://zhuanlan.zhihu.com/p/168742724
https://blog.csdn.net/qq_30091945/

![](https://img-blog.csdnimg.cn/78c89ca6b1544b67ac297b16294d232a.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6L-b6Zi25aqb5bCP5ZC0,size_20,color_FFFFFF,t_70,g_se,x_16)

