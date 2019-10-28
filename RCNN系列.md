# RCNN(classification+regression)
### 过程:
- 训练分类网络
- 模型做fine-tuning

  类别1000改为20，加上背景21
  
  去掉fc
- 特征提取
  
  提取候选框（选择性搜索）
  
  (1)生成区域集R  Efficient Graph Based Image Segmentation 
 
  (2)计算区域集R里每个相邻区域的相似度S={s1,s2,…}
  
  (3)找出相似度最高的两个区域，将其合并为新的集合，添加进R
  
  (4)从S中移除所有与step2中有关的子集
  
  (5)计算新集与所有子集的相似度
  
  (6)跳至(3)直到S为空
- 训练SVM分类器，每个类别对应一个SVM

- 回归器精修候选框位置： 利用线性回归模型判定框的准确度

### shortness:
- 候选框选择算法耗时严重 Object detection is slow.
- 重叠区域特征重复计算
- Training is a multi-stage pipeline.
- Training is expensive in space and time.

# FastRCNN(RPIPooing)

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/fastrcnn.png)

### Improve:
- Training is single-stage, using a multi-task loss
- Training can update all network layers
- No disk storage is required for feature caching

### Three transformations for pre-trained networks:
- The last max pooling layer is replaced by a RoI pooling layer that is conﬁgured by setting H and W
- fc:K +1
- The network is modified to take two data inputs: a list of images and a list of RoIs in those images.

### Loss function

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/fastrcnn_loss.png)

### ROIPooling
The RoI pooling layer uses max pooling to convert the features inside any valid region of interest into a small feature map with a ﬁxed spatial extent of H ×W(hyper-parameters)

为了将proposal抠出来的过程，然后resize到统一的大小。
- 根据输入的image，将ROI映射到feature map对应的位置
- 将映射后的区域划分为相同大小的sections
- 对每个section进行max pooling操作

### shortness:
- selective box，找出所有的候选框十分耗时

# FasterRCNN(RPN)

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/fasterrcnn.png)

### Region proposal network

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/fasterrcnn_rpn.png)
- This architecture is naturally implemented with an n×n convolutional layer followed by two sibling 1×1 convolutional layers.
- 端到端的检测
- enabling nearly cost-free region proposals
- computing proposals with a deep convolutional neural network
- On top of these convolutional features, we construct an RPN by adding a few additional 
convolutional layers that simultaneously regress region bounds and objectness scores at each location on a regular grid
introduce novel “anchor” boxes that serve as references at multiple scales and aspect ratios.

### Anchor
- 前景背景分类+框的位置的回归
- 三个面积尺寸(128 256 512)
- 在每个面积尺寸下，取三种不同的长宽比例(1：1, 1:2, 2:1)
- K=9

### Loss function

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/fasterrcnn_loss.png)

# MaskRCNN(RPNAlign)

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/maskrcnn.png)
- Extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.
- The convolutional backbone architecture used for feature extraction over an entire image.
- The network head for bounding-box recognition(classification and regression) and mask prediction that is applied separately to each RoI.

### RPNAlign
- It improves mask accuracy by relative 10% to 50%, showing bigger gains under stricter localization metrics. 
- Essential to decouple mask and class prediction.
- We avoid any quantization of the RoI boundaries or bins(i.e.,we use x/16 instead of [x/16]).

### Loss function

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/maskrcnn_loss.png)

- The mask branch has a K×m×m dimensional output for each RoI, which encodes K binary masks of resolution m×m, one for each of the K classes. 
- To this we apply a per-pixel sigmoid, and deﬁne Lmask as the average binary cross-entropy loss. 
- For an RoI associated with ground-truth class k, Lmask is only deﬁned on the k-th mask (other mask outputs do not contribute to the loss).

# FasterRCNN 代码复现(部分笔记,未完全看完代码)

### 环境搭建和代码运行
```
conda create -n fasterrcnn-pt python=3.6.9
conda activate fasterrcnn-pt

conda install pytorch=0.4.0 torchvision cuda90 -c pytorch
pip install opencv-python easydict tensorboard-pytorch 
pip install scipy pycocotools matplotlib pillow torchvision
pip install cython tensorflow

# next:https://github.com/ruotianluo/pytorch-faster-rcnn/tree/0.4#Installation
```
```
# 训练网络
python ./tools/trainval_net.py --weight data/imagenet_weights/res101.pth 
--imdb voc_2007_trainval 
--imdbval voc_2007_test --iters 70000 --cfg experiments/cfgs/res101.yml  
--tag  experiments/logs/res101_voc07/ --net res101 --set ANCHOR_SCALES [8,16,32] 
ANCHOR_RATIOS [0.5,1,2] TRAIN.STEPSIZE [50000]

# 测试网络
python ./tools/test_net.py --imdb voc_2007_test 
--model output/mobile/voc_2007_trainval/experiments/logs/mobile/
mobile_faster_rcnn_iter_70000.pth 
--cfg experiments/cfgs/mobile.yml  --tag  experiments/logs/mobile/ --net mobile 
--set ANCHOR_SCALES [8,16,32] ANCHOR_RATIOS [0.5,1,2]

# 可视化
tensorboard --logdir=tensorboard/mobile/voc_2007_trainval/ --port=7001
```

### 代码分析
- 训练网络必须有一个网络(继承nn.Module)和一个数据生成器(eg.RoIDataLayer)
- 数据roidb的格式
```
{
    'width'
    'height'
    'boxes': [[xmin, ymin, xmax, ymax],...]
    'gt_classes': [[cls1],...]
    'gt_overlaps': -1.0 or 1.0
    'flipped': False/True 是否翻转
    'seg_areas': 
    'max_overlaps':
    'max_classes':
}
```
- 网络结构
  
  ![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/code_rpn.png)
 
- Generate_anchors

  ![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/code_generate_anchors.png)
   
- Loss function 
  
  ![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/code_rpn_loss.png)

### 实验细节
- 数据处理部分对train部分的图片进行了翻转处理，val部分图片不做翻转处理
- 数据处理对overlap作了筛选
- 网络部分pytorch封装了resnet,vgg等常见网络，在训练的时候直接调用但是注意固定参数训练只做模型参数的微调和去掉不需要的部分网络
- 训练中学习率有一次衰减
- 生成anchor的base_size fixes to 16

### 实验参数具体设置
- 图像均值是一个超参数
- RoIDataLayer每次只向网络注入一张图片cfg.TRAIN.IMS_PER_BATCH=1


### Question
- 学习率衰减的步数如何能达到较好的效果
- RPN和ROI网络能否进一步优化
- 数据预处理部分如何处理大小不一致的图片
