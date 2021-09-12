 [简体中文](README.md) | English
# CV-Paper

This program is mainly for novice learners and self-review. What makes a classic a classic is its unique ideas and substantial improvement in quality. This project is mainly about papers in CV direction, and there may be some articles in NLP direction. The following are some of my personal experiences, for reference only.

Continuously updated ......

## Introduction
### Required Content -- Books
1. [Machine Learning](./book)" by Mr. Zhou Zhihua, "Watermelon Book". Formula derivation companion using ["Pumpkin Book"](https://datawhalechina.github.io/pumpkin-book/#/)

2. ["《"Statistical learning methods"](./book)--Mr. Li Hang.

3. " Flower book "[-deep learning](./book)--The 2018 Turing Award dive into the AI bible.

4. ["Dive into Deep Learning"](https://zh-v2.d2l.ai/)--Mr. Li Mu (Mu Shen) of Amazon team make deep learning approachable, teaching you the concepts, the context, and the code.

###  Required Content -- Video

1. python recommends by Zhou Morfan [mofan python](https://mofanpy.com/)。python basics is great for people who are just starting out, or who have used the language before, each video is not too long, and the sections are connected, so it's great for getting up to speed on the basics. A great guy who quietly dedicates himself to sharing what he has learned.


2. Hongyi Li's course on deep learning and machine learning. Li Hongyi's video is very, very good and detailed, I am very amazed that someone can give such a good lecture.[Machine Learning](https://www.bilibili.com/video/BV1Wv411h7kN?from=search&seid=12925803382584433478&spm_id_from=333.337.0.0)，[Deep Learning](https://www.bilibili.com/video/BV1JE411g7XF?from=search&seid=12925803382584433478&spm_id_from=333.337.0.0)


3. Mr. Hu Haoji, Zhejiang University[Graduate Machine Learning Program](https://www.bilibili.com/video/BV1dJ411B7gh). Human handwritten board, and speaks really well, mathematical derivation process speaks clearly. It can be consumed together with Li Hang's "Statistical Learning Methods". Many of the derivations not clearly explained in the book, listened to Mr. Hu's explanation after a clear understanding. A teacher who lectures until his body is wet.


4. Li Feifei's  [csn231 Course](https://www.bilibili.com/video/BV1nJ411z7fe?from=search&seid=6244444160908607298&spm_id_from=333.337.0.0)。I haven't watched this course in its entirety. But it is an introductory video that many people recommend as a must-see.


5. Andrew Ng[Machine Learning](https://www.bilibili.com/video/BV1VK4y1d7fj). For those who have a good English background, you can watch the full English video of Andrew Ng.




* [Image classification](#Imageclassification)
* [Object detection](#Object detection)
* [Image Segmentation](#Image Segmentation
* [Medical imaging-related](#Medical imaging-related)
* [Attention mechanism in CV](#Attention mechanism in CV)
* [Others(#Others)
* [Data Sets](#Data Sets)



## Image-classification
- [LeNet](https://ieeexplore.ieee.org/document/726791)

 In 1998, "Gradient-Based Learning Applied to Document Recognition" was the pioneering work of CNN and the classic paper on handwriting recognition. Any network that contains a convolutional layer can be understood as a convolutional neural network.

- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

 2012 "ImageNet Classification with Deep Convolutional Neural Networks" ILSVRC-2012 contest winner, promoting the CNN's masterpiece, AlexNet is a historic turn in the history of CNN development, can not be unread.

 [Personal understanding](/notes/AlexNet.md)

- [FZNet](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)

 There is still little insight into the inner workings and behavior of AlexNet's complex models, or how they achieve such good performance. You are so good, and I still don't know why you are so good. Well, I have to get to know you a little bit.「🤔」
 [Personal understanding](/notes/FZNet.md)

- [ZFNet](https://arxiv.org/abs/1311.2901)



- [VGGNet](https://arxiv.org/abs/1409.1556)

"Very Deep Convolutional Networks for Large-Scale Image Recognition", although not the winner of that year's ImageNet competition (the winner of that year was GoogLeNet), VGGNet had an important impact on the later ResNet and Inception.
 [Personal understanding](/notes/VGG.md)

- [GoogleNet](https://arxiv.org/abs/1409.4842)和[Inception V3](https://arxiv.org/abs/1512.00567)

 GoogleNet（Inception V1）, Inception V3:《Going Deeper with Convolutions》,《Rethinking the Inception Architecture for Computer Vision》,Winner of ImageNet 2014, the Inception structure is cleverly designed .
 
  [Personal understanding](/notes/GoogleNet.md)
 
- [ResNet](https://arxiv.org/abs/1512.03385)

 "Deep Residual Learning for Image Recognition", which directly reduces the top5 error rate to 3.57% (GoogLeNet is 6.66%), surpassing the human eye, and the biggest highlight in the paper is the design of the residual block structure.
 
  [Personal understanding(/notes/ResNet和ResNeXt.md)
 
- [ResNeXt](https://arxiv.org/abs/1611.05431)

 [Personal understanding](/notes/ResNet和ResNeXt.md)


* [Inception-v4, Inception ResNet](https://arxiv.org/abs/1602.07261)
    *  [Inception-v4 Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py)
    *  [Inception-Resnet-V2 Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py)


- [MobileNet v1](https://arxiv.org/abs/1704.04861)


- [MobileNet v2](https://arxiv.org/abs/1704.04861)


- [MobileNet v3](https://arxiv.org/abs/1905.02244)


- [ShuffleNet v1](https://arxiv.org/abs/1707.01083)


- [ShuffleNet v2](https://arxiv.org/abs/1807.11164v1)


- [EfficientNet_v1](https://arxiv.org/abs/1905.11946v3)


- [EfficientNet_v2](https://arxiv.org/abs/2104.00298)


- [SENet](https://arxiv.org/abs/1709.01507)



## Object detection
 The RCNN family includes RCNN, Fast RCNN, Faster RCNN, Mask RCNN

YOLO series includes YOLO v1, YOLO v2, YOLO v3 and SSD


- [RNN](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)


- [RCNN](https://arxiv.org/pdf/1311.2524v5.pdf)

  [Personal understanding](/notes/RCNN.md)



- [Fast-RCNN](https://arxiv.org/abs/1504.08083)


- [Faster-RCNN](https://arxiv.org/abs/1506.01497)

Some disadvantages and shortcomings of RCNN and SPPnet:

- **The training process is a multi-stage pipeline**. The RCNN is first fine-tuned using log loss on a given region proposal. Then the features extracted from the convolutional neural network are trained to the SVM classifier, using SVM instead of softmax which is commonly used in neural network classification algorithms. the third part is the regression of the learning detection frame.

- **Training requires a lot of space and time**， Since the training process requires writing the features extracted by the convolutional neural network to disk, a large amount of physical storage space is required and the training process is very slow.
- **Very slow detection process**，At test time, features are extracted from each target candidate frame in each test image.
RCNN mainly inputs each regional proposal of each image into the CNN" network for computation, without and retrograde corresponding shared computation, while SPPnet uses shared convolutional computation to accelerate the detection process of RCNN, SPPnet inputs the whole image into the CNN network to get the feature map, and then uses the spatial pyramid pooling network to process the feature map of each regional proposal region to get the fixed dimensional feature vector, and then trains the SVM classifier.


To address these advantages, the main contributions of Fast RCNN are:

     1.Fast RCNN has higher accuracy of target detection    
     2.The training process uses a multi-task loss function
     3.Training can update the parameters of all network layers
     4.No additional disk space required to store features


- [FPN](https://arxiv.org/abs/1612.03144)


- [SSD](https://arxiv.org/abs/1512.02325)


- [RetinaNet](https://arxiv.org/abs/1708.02002)

RetinaNet is a target detection model before the birth of YOLO V3, following the announcement of SSD and YOLO V2, from "Focal Loss for Dense Object Detection" by Kai-Ming He, the great god. The full paper addresses the imbalance between the foreground (positive) and background (negatives) categories in existing one-stage object detection models, and proposes a loss function called Focal Loss, which is used to reduce the weight of a large number of easy negatives in the standard cross-entropy (and increase the weight of hard negatives). negatives). In order to test the effectiveness of the proposed Focal Loss loss function, so the authors then proposed a simple model RetinaNet in passing.



- [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)


- [SPPNet](https://arxiv.org/abs/1406.4729)

github(PyTorch): https://github.com/yueruchen/sppnet-pytorch

Spatial pyramidal pooling in deep convolutional neural networks for visual recognition. the proposed SPP-Net solves the problem of fixed input image size on the one hand, i.e., the input image size is not limited. Because SPP-Net solves the problem that the input of fully connected layer needs fixed dimensionality.

 why do the inputs to the fully connected layer need to be fixed dimensional?

A: The computation of the fully connected layer is actually equivalent to inner-product of the input feature map data matrix and the fully connected layer weight matrix. When configuring a network, the parameter dimension of the fully connected layer is fixed, so for the two matrices to be able to perform inner product, the dimension of the input feature map data matrix also needs to be fixed [X1][X2].


The shortcomings of RCNN are:
- Each image extracts about 2,000 candidate regions (Region Proposal), and each Region Proposal uses CNN repeatedly to extract features, so a lot of time is spent in the feature extraction phase.
- Since the input dimension of the fully connected layer is fixed, it must be resized (crop/wrap) for Region Proposal to perform feature extraction, which results in loss of image information and image deformation and affects the accuracy of feature extraction.

So SPPNet has been improved for two shortcomings of R-CNN：
- Instead of putting the Region Proposal of Selective Search into CNN for feature extraction, the original image is directly put into CNN for feature extraction, and then a feature map is made in conv5 according to the location of Region Proposal, and then the feature map mapped by each Region Proposal is truncated. This avoids the repetitive use of CNN to extract features for each Region Proposal individually, which reduces a lot of time.
- SPPNet adds Spatial Pyramid Pooling layer after the original CNN's conv5 to replace the original Pooling5 layer, which can accept feature maps of different sizes and output feature maps of the same size, thus avoiding the image deformation caused by resize.
　　To summarize, the original intention of SPP-net is very clear, that is, we hope that the network is more flexible to the size of the input, the analysis to the convolutional network does not require the size, the requirement of fixed size comes entirely from the fully connected layer part, and thus with the help of spatial pyramid pooling to bridge the two, SPP-net in the field of detection is an important contribution to avoid the deformation of R-CNN, repeated computation and other problems, in the case of the effect does not decay, significantly improve the recognition speed.


## 图像分割
- [FCN](https://arxiv.org/abs/1411.4038)

（CVPR 2015 best paper honorable mention）If you know classification networks, then you will understand segmentation networks quickly, because classification networks predict a picture into a class, while segmentation networks predict every pixel of a picture, which is actually the same.

- [R-FCN](https://arxiv.org/abs/1605.06409)

- DeepLab Series: [DeepLab V1](https://arxiv.org/abs/1412.7062v4)、[DeepLab V2](https://arxiv.org/abs/1606.00915)、[DeepLab V3](https://arxiv.org/abs/1706.05587)和[DeepLab V3+](https://arxiv.org/abs/1802.02611)，

 [Personal understanding](/notes/DeepLab.md)

## 医学影像相关
[DeepLung](https://arxiv.org/abs/1801.09555)

Project ：https://github.com/wentaozhu/DeepLung、https://github.com/uci-cbcl/DeepLung

 A fully automated lung CT cancer diagnosis system DeepLung is proposed.DeepLung consists of two parts, nodule detection (identifying candidate nodule locations) and classification (classifying candidate nodules as benign or malignant). Considering the 3D nature of lung CT data and the compressibility of dual path network (DPN), two deep 3D DPNs were designed for nodule detection and regression, respectively. Specifically, a Faster RCNN with 3D dual-path blocks and U-net type encoding-decoding structure is used to learn nodule features efficiently. For nodule classification, a gradient boosting machine (GBM) with 3D dual-path network features is proposed. The nodule classification subnetwork is validated on a public dataset complex of LIDC-IDRI, achieving better performance than state-of-the-art and outperforming experienced doctors on image-based patterns. In the DeepLung system, candidate nodules are first detected by the nodule detection subnetwork, and then the nodule diagnosis is done using the classification subnetwork. Extensive experimental results show that DeepLung performs comparably to experienced physicians for both node-level and patient-level diagnosis on the LIDC-IDRI dataset.


## CV中的Attention机制
- [Transformer](https://arxiv.org/abs/1706.03762)

 Transformer was proposed by the paper "Attention is All You Need" and is now the recommended reference model for Google Cloud TPU.

**Motivation**:
1. Relying on attention mechanism, not using RNNand CNN, high parallelism
2. Catching long-range dependencies by attention is stronger than RNN

**Innovation**：

1. By self-attention, one does attention with oneself, so that each word has global semantic information (long dependency)

2. Since Self-Attention computes Attention for every word and all words, the maximum path length is only 1, no matter how long the distance between them. long-distance dependencies can be captured.

3. Multi-head attention is proposed, which can be seen as an ensemble version of attention, where different heads learn different subspace semantics.

- [Vision Transformer(ViT)](https://openreview.net/pdf?id=YicbFdNTTy)

1. The authors try to apply the Transformer structure directly to an image by segmenting an image into multiple patches, which are regarded as tokens (words) of NLP, and then doing a series of linear embedding operations on each patches as the input of the Transformer.
2. However, the authors found that the method trained on mid-sized datasets like ImageNet, acc is a little lower than ResNets of the same size. The reason may be that Transformers lack the inherent inductive biases of CNNs, such as translation invariance and locality, and thus do not generalize well when the training set is not sufficient.
 
3. However, the authors found that when pretraining on a larger dataset and then transferring to other tasks, the Transformer performed better.

 [Personal understanding](/notes/ViT.md)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)

代码：[Semantic segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)，[Image classification](https://github.com/microsoft/Swin-Transformer)，[Object detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)

- [BoTNet](https://arxiv.org/abs/2101.11605)

The architecture incorporates self-attention for a variety of computer vision tasks, including image classification, target detection, and instance segmentation. By replacing the spatial convolution with global self-attention in only the last three bottleneck blocks of ResNet and making no other changes, our approach significantly improves the baseline in instance segmentation and target detection, while also reducing parameters to minimize latency.
 
 [Personal understanding](/notes/BoTNet.md)

## Others

- [DeepID2+](https://arxiv.org/abs/1412.1265)

“Deeply learned face representations are sparse, selective, and robust”  Why do we recommend this paper? In the field of face recognition, DeepID is as famous as ever. Unlike DeepID and DeepID2, this paper is not just about face recognition, it analyzes the internal structure of CNN and tries to explain the powerful feature extraction and classification recognition ability of CNN from the theoretical point of view. After reading this paper, I believe there will be a more in-depth understanding of CNN.

- [GPT-3](https://arxiv.org/abs/2005.14165)

 github：https://github.com/openai/gpt-3

For all tasks, applying GPT-3 does not require any gradient updating or fine-tuning, but simply specifying the task and a small number of presentations by interacting with the model's text. GPT-3 has excellent performance on many NLP datasets, including translation, question-answering and completion tasks, as well as some tasks that require on-the-fly inference or domain adaptation.


## Data Sets

* [ImageNet](#ImageNett)
* [COCO](#COCO)
* [CIFAR-10](#CIFAR-10)
* [PASCAL VOC](#PASCAL-VOC)

### [ImageNet](https://paperswithcode.com/dataset/imagenet)

 The ImageNet dataset contains 14,197,122 annotated images according to the WordNet hierarchy. Since 2010 the dataset is used in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), a benchmark in image classification and object detection. The publicly released dataset contains a set of manually annotated training images. A set of test images is also released, with the manual annotations withheld. ILSVRC annotations fall into one of two categories: (1) image-level annotation of a binary label for the presence or absence of an object class in the image, e.g., “there are cars in this image” but “there are no tigers,” and (2) object-level annotation of a tight bounding box and class label around an object instance in the image, e.g., “there is a screwdriver centered at position (20,25) with width of 50 pixels and height of 30 pixels”. The ImageNet project does not own the copyright of the images, therefore only thumbnails and URLs of images are provided.
 
- Total number of non-empty WordNet synsets: 21841
-  Total number of images: 14197122
- Number of images with bounding box annotations: 1,034,908
- Number of synsets with SIFT features: 1000
- Number of images with SIFT features: 1.2 million

### [COCO](https://paperswithcode.com/dataset/coco)
 The MS COCO (Microsoft Common Objects in Context) dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images.

 Splits: The first version of MS COCO dataset was released in 2014. It contains 164K images split into training (83K), validation (41K) and test (41K) sets. In 2015 additional test set of 81K images was released, including all the previous test images and 40K new images.

 Based on community feedback, in 2017 the training/validation split was changed from 83K/41K to 118K/5K. The new split uses the same images and annotations. The 2017 test set is a subset of 41K images of the 2015 test set. Additionally, the 2017 release contains a new unannotated dataset of 123K images.
 
 Annotations: The dataset has annotations for

 - object detection: bounding boxes and per-instance segmentation masks with 80 object categories,
 - captioning: natural language descriptions of the images (see MS COCO Captions),
-  keypoints detection: containing more than 200,000 images and 250,000 person instances labeled with keypoints (17 possible keypoints, such as left eye, nose, right hip, right ankle),
 - stuff image segmentation – per-pixel segmentation masks with 91 stuff categories, such as grass, wall, sky (see MS COCO Stuff),
 - panoptic: full scene segmentation, with 80 thing categories (such as person, bicycle, elephant) and a subset of 91 stuff categories (grass, sky, road),
 - dense pose: more than 39,000 images and 56,000 person instances labeled with DensePose annotations – each labeled person is annotated with an instance id and a mapping between image pixels that belong to that person body and a template 3D model. The annotations are publicly available only for training and validation images.

### CIFAR-10

 The CIFAR-10 dataset (Canadian Institute for Advanced Research, 10 classes) is a subset of the Tiny Images dataset and consists of 60000 32x32 color images. The images are labelled with one of 10 mutually exclusive classes: airplane, automobile (but not truck or pickup truck), bird, cat, deer, dog, frog, horse, ship, and truck (but not pickup truck). There are 6000 images per class with 5000 training and 1000 testing images per class.

 The criteria for deciding whether an image belongs to a class were as follows:

- The class name should be high on the list of likely answers to the question “What is in this picture?”
- The image should be photo-realistic. Labelers were instructed to reject line drawings.
- The image should contain only one prominent instance of the object to which the class refers. The object may be partially occluded or seen from an unusual viewpoint as long as its identity is still clear to the labeler.

### PASCAL-VOC

PASCAL VOC: The dataset is a relatively well-known dataset in the field of target detection, which is divided into two subsets, VOC2007 and VOC2012, and its official download address is as follows：
* [VOC2007-trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
* [VOC2007-test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
* [VOC2012-trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
Official website is often unstable, which leads to slow download or even download failure, we recommend a certain stable mirror source address：[YOLOv3 Author's Home Page](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
