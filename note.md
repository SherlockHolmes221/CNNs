# Return of the Devil in the Details: Delving Deep into Convolutional Nets(2014)
#### 主要是CNN和IFV为代表的浅层特征表示方法对图像分类效果的对比
#### 以及图像处理细节(colour information, feature normalisation, and data augmentation)带来的影响

##### 主要结论:
- CNN-based methods consistently outperform the shallow encodings.
- The CNN output layer can be reduced significantly without having an adverse effect on performance.
- The performance of shallow representations can be significantly improved by adopting data augmentation, typically used in deep learning.
- L2-normalising the CNN features before use in the SVM was found to be important for performance.
- The performance of deep representations on the ILSVRC dataset is a good indicator of their performance on other datasets, 
- Fine-tuning can further improve on already very strong results achieved using the combination of deep representations and a linear SVM.

##### 实验细节:
- 实验的网络结构

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/devil_arch.png)
(fc with dropout!!)
- 数据集 voc2007(mAP) voc2012 ILSVRC-2012(top-5) Caltech-101  Caltech-256
- Training
 
  dataset: ILSVRC-2012
  
  momentum 0.9,weight decay 5·10−4
  
  initial learning rate 10−2 decreased by a factor of 10 when the validation error stop decreasing
  
  The layers are initialised from a Gaussian distribution with a zero mean and variance equal to 10−2
- The loss function of voc

  one-vs-rest classiﬁcation hinge loss 
  
  ranking hinge loss
  
##### 实验结果和对比结论:

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/devil_result.png)

- Ablation experiments1:data augmentation

  no augmentation,random crops, horizontal ﬂips, and RGB colour jittering

  Augmentation consistently improves performance by ∼ 3% for both IFV and CNN 
  
  Fipping improves only marginally.
  
  The more expensive C+F sampling improves, as seen, by about 2 ∼ 3%.
- Ablation experiments2: CNN ﬁne-tuning vs without ﬁne-tuning 
  
  Fine-tuning is able to adjust the learnt deep representation to better suit the dataset in question.
- Ablation experiments3:different loss functions of voc
  
  On VOC-2012, using the ranking loss is marginally better than the classification loss , which can be explained by the ranking-based VOC evaluation criterion.
- Ablation experiments4:different dimensions of fc layers

   We can reduce the output dimensionality further to 1024D and even 128D with only a drop of ∼ 2% for codes that are 32× smaller.
- Ablation experiments5:Colour information

  SIFT and colour descriptors are combined by stacking the corresponding IFVs,there is a small but significant improvement of around ∼ 1% in the non-augmented case. 
   
  Retraining the network after converting all the input images to grayscale (denoted GS in Methods) has a more significant impact, resulting in a performance drop of ∼ 3%.
- Ablation experiments6:different architectures(3 kinds of CNNs and IFV)

  Both medium CNN-M and slow CNN-S outperform the fast CNN-F by a significant 2 ∼ 3% margin.
  
  An advantage of CNNs compared to IFV is the small dimensionality of the output features.
  
##### something questions:
- How data augmentation samples used sum/max-pooling or stacking
- How they choose CNN arch
- The details of loss function of voc
- IFV is hard to understand.

# Network In Network(2014) based on maxout network

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/nin.png)

NINs include the stacking of three mlpconv layers and one global average pooling layer

#### NIN:
 This structure consists of mlpconv layers which use multilayer perceptrons to convolve the input 
 and a global average pooling layer as a replacement for the fully connected layers in conventional CNN.
#### The shortness of convolutional filter and fully connected layers
- abstraction is low with generalized linear model (GLM)
- The data for the same concept often live on a nonlinear manifold.(how to comprehend??)
- The fully connected layers are prone to overfitting and heavily depend on dropout regularization.

#### What is MLP Convolution Layers

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/nin_formula.png)

1x1 convolution kernel and cascaded cross channel parametric pooling on a normal convolution layer

- The mlpconv maps the input local patch to the output feature vector with a multi 
  layer perceptron (MLP) consisting of multiple fully connected layers with nonlinear activation functions. 

#### Why micro MLP Convolution Layers works?
- It is a general nonlinear function approximator.
- Multilayer perceptron can be a deep model,which is consistent with the spirit of feature re-use,不用像CNN那样靠卷积层的堆叠得到high level feature

#### Why global average pooling works?
- global average pooling is more meaningful and interpretable as it enforces correspondance between feature maps and categories
  instead of black box as fully connection layers.
  ( global average pooling over the fully connected layers is that it is more native to the convolution structure by 
  enforcing correspondences between feature maps and categories. )
- Global average pooling is itself a structural regularizer, which natively prevents overfitting for the overall structure.
 (There is no parameter to optimize in the global average pooling thus overfitting is avoided at this layer. )
- Global average pooling sums out the spatial information, thus it is more robust to spatial translations of the input. 

#### something questions:
- Weather it works well when the mlpconv goes deeper?
- How to comprehend:
  Mlpconv layer differs from maxout layer in that the convex function approximator is replaced by a universal function approximator, which has greater capability in modeling various distributions of latent concepts.
- The details of data augmentation is not mentioned in this paper.

# Identity Mappings in Deep Residual Networks
 2 improvements: 
- Identity shortcut connections 
- Identity after-addition activation
 
 ![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/resnet_compare.png)
 
 origin vs proposed:
 
 <figure class="half">
    <img src="https://github.com/SherlockHolmes221/CNNs/raw/master/img/res_origin_formula.png">
    <img src="https://github.com/SherlockHolmes221/CNNs/raw/master/img/resnet_fproposed_formula.png">
</figure>

result:

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/resnet_result.png)

#### Why it works
- The forward and backward signals can be directly propagated from one block to any other block

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/resnet_backward.png)

- The gradient of a layer does not vanish even when the weights are arbitrarily small.
- Ease of optimization：
  
   Using the original design, the training error is reduced very slowly at the beginning of training. 
   
   For f = ReLU, the signal is impacted if it is negative, and when there are many Residual Units, this effect becomes prominent.
   
   When f is an identity mapping, the signal can be propagated directly between any two units. 
- Reducing overfitting
  
  The pre-activation version reaches slightly higher training loss at convergence, but produces lower test error.
  
  This is presumably caused by BN’s regularization effect.
  
  In the original Residual Unit, although the BN normalizes the signal, this is soon added to the shortcut and thus the merged signal is not normalized. 

#### Some ablation experiments on shortcut connections:

conclusion:the Shortcut connections are the most direct paths for the information to propagate.

Multiplicative manipulations (scaling, gating, 1×1 convolutions, and dropout) on the shortcuts can hamper information propagation and lead to optimization problems.
- Constant scaling

  Suggesting that the optimization has difficulties when the shortcut signal is scaled down.
- Exclusive gating
   
  The impact of the exclusive gating mechanism is two-fold.
- 1×1 convolutional shortcut
  
  When stacking so many Residual Units (54 for ResNet-110), even the shortest path may still impede signal propagation.
   
- Dropout shortcut  
  
  The network fails to converge to a good solution.

#### Some ablation experiments on activation functions

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/resnet_experiment.png)

- BN after addition
  
  The results become considerably worse than the baseline.
  
  The BN layer alters the signal that passes through the shortcut and impedes information propagation, 
  as reflected by the difficulties on reducing training loss at the beginning of training.
- ReLU before addition

   As a result, the forward propagated signal is monotonically increasing. 
   This may impact the representational ability, and the result is worse.
- Post-activation or pre-activation

   The ReLU-only pre-activation performs very similar to the baseline on ResNet-110/164. 
   
   This ReLU layer is not used in conjunction with a BN layer, and may not enjoy the benefits of BN.
   
   When BN and ReLU are both used as pre-activation, the results are improved by healthy margins.
   
   
