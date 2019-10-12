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

# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION(2015)
为了增加网络的深度 use very small (3×3) convolution filters, small receptive field
A stack of two 3×3 conv layers (without spatial pooling in between) has an effective receptive field of 5×5.

#### why CNNs enjoy great success?
- The large public image repositories
- High-performance computing systems

#### Architecture details

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/vgg.png)

- Use filters with a very small receptive field: 3 × 3 
- In one of the configurations utilise 1 × 1 convolution filters, which can be seen as a linear transformation of the input channels.
- Spatial pooling is carried out by ﬁve max-pooling layers, which follow some of the conv.
- A stack of convolutional layers (which has a different depth in different architectures) is followed by three Fully-Connected (FC) layers
- All hidden layers are equipped with the rectification (ReLU)
- Local Response Normalisation does not improve the performance on the ILSVRC dataset.(why??)

#### why it works:
- makes the decision function more discriminative
- decrease the number of parameters
- The incorporation of 1 × 1 conv layers is a way to increase the nonlinearity of the decision function without affecting the receptive fields of the conv layers

#### Experiment details 实验细节中对数据的预处理做得比较好
- Data augmentation: 
  
  pixel subtracting the mean RGB value, random horizontal flipping and random RGB colour shift
  
  randomly cropped from rescaled training images (one crop per image per SGD iteration)

- mini-batch 256
- SGD(momentum 0.9  weight decay(L2 5*10^-4))
- learning rate was initially set to 10−2, and then decreased by a factor of 10 when the validation set accuracy stopped improving
- dropout 0.5

#### Experiment 
- Local response normalisation does not improve on the model A without any normalisation layers.  
- Classification error decreases with the increased ConvNet depth.
- A deep net with small filters outperforms a shallow net with larger filters.

# Going Deeper with Convolutions(2015) Inception_v1

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv1.png)

Introduce sparsity and replace the fully connected layers by the sparse ones 

Increased the depth and width of the network while keeping the computational budget constant.

#### Architecture details

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv1_arch.png)

- In order to avoid patch-alignment issues, 
current incarnations of the Inception architecture are restricted to filter sizes 1×1, 3×3 and 5×5
- The suggested architecture is a combination of all those layers with their output filter 
banks concatenated into a single output vector forming the input of the next stage.
- The ratio of 3×3 and 5×5 convolutions should increase as we move to higher layers.
- Judiciously reducing dimension wherever the computational requirements would increase too much otherwise.
- All the convolutions, including those inside the Inception modules, use rectified linear activation.
- Average pooling before the classifier

#### Why it works:
- sparse structure
- combination of all those layers with their output filter 
- adding an alternative parallel pooling path 
- 1 * 1:降维  构成了更有效的卷积层

#### Experiment details
- SGD(momentum=0.9)
- Fixed learning rate schedule(decreasing the learning rate by 4% every 8 epochs)
- Trained 7 versions of the same GoogLeNet model (including one wider version), and performed ensemble prediction.
- Testing:144-crops

#### Experiment result

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv1_retult.png)

# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift(2015) Inception_v2

#### What is Batch Normalization:

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/bn.png)

#### What problems Batch Normalization solves:
-  internal covariate 
   distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. 
   This slows down the training by requiring lower learning rates and careful parameter initialization, 
   and makes it notoriously hard to train models with saturating nonlinearities. 
- 
#### Why Batch Normalization works:
- Use much higher learningrates and be less careful about initialization.
- Acts as a regularizer, regularizes the model and reduces the need for Dropout
- Makes it possible to use saturating nonlinearities by preventing the network from getting stuck in the saturated modes
- Batch Normalization enables higher learning rates
- Batch Normalization regularizes the model

#### Experiments:
- By only using Batch Normalization ( BN-Baseline ), we match the accuracy of Inception in less than half the number of training steps.
- By applying the modifications(Increase learning rate, Remove Dropout, Reduce the photometric distortions,
  Reduce the L2 weight regularization, Accelerate the learning rate decay, Shuffle training examples more thoroughly), 
  we significantly increase the training speed of the network.

# Rethinking the Inception Architecture for ComputerVision(2015) (a little hard to understand)
improve computational efficiency and reduce low parameter counts

#### Architecture

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv3.png)

#### Some Design Principles
- Avoid representational bottlenecks, especially early in the network.
- Higher dimensional representations are easier to process locally within a network
- Spatial aggregation can be done over lower dimensional embeddings 
  without much or any loss in representational power.
- Balance the width and depth of the network

#### Improve points
- Factorization into smaller convolutions 5×5conv = 2×(3×3conv) 7×7conv = 3×(3×3conv) 

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv3_1.png)

 <figure class="half">
    <img src="https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv3_2.png">
    <img src="https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv3_3.png">
</figure>

- Spatial Factorization into Asymmetric Convolutions 

 <figure class="half">
    <img src="https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv3_4.png">
    <img src="https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv3_5.png">
</figure>

- Efficient Grid Size Reduction

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv3_6.png)

#### Experiment details
- batch size 32 for 100 epochs,momentum=0.9
-  RMSProp with decay of 0.9 and $\in$= 1.0.
- learning rate of 0.045, decayed every two epoch using an exponential rate of 0.94. 

#### Experiment result

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv3_7.png)

#### Some questions
- What is the meaning of 'Higher dimensional representations are easier to process locally within a network'
- why high quality results can be reached with receptive field resolution as low as 79×79. 
  This might prove to be helpful in systems for detecting relatively small objects.
# Deep Residual Learning for Image Recognition(2015)

 <figure class="half">
    <img src="https://github.com/SherlockHolmes221/CNNs/raw/master/img/resnet1.png">
    <img src="https://github.com/SherlockHolmes221/CNNs/raw/master/img/resnet_formula.png">
</figure>

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/resnet2.png)

#### Design rules:
- For the same output feature map size, the layers have the same number of filters
- If the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer.
- conv-bn-relu

#### Some shortness of deeper networks： 
- 深层的网络梯度容易消失(can solved by BN)
- With the network depth increasing, accuracy gets saturated and then degrades rapidly.

#### Resnet improves:
- Ease the training of networks that are substantially deeper than those used previously
- 更快收敛 Our extremely deep residual nets are easy to optimize, but the counterpart “plain” nets (that simply stack layers) exhibit higher training error when the depth increases
- 误差更小 Our deep residual nets can easily enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks

#### Experiment details
- Data augmentation: randomly sampled, scale augmentation, per-pixel mean subtracted
- SGD(lr=0.1, divided by 10 when the error plateaus, weight decay of 0.0001 and a momentum of 0.9. )
- mini-batch size of 256
- epoch= 60×10^4 
- without dropout 
- test: standard 10-crop testing, average the scores at multiple scales

# Identity Mappings in Deep Residual Networks(2016)
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
   
# Inception-v4,Inception-ResNet and the Impact of Residual Connections on Learning(2016)
combine the Inception architecture with residual connections.

 <figure class="half">
    <img src="https://github.com/SherlockHolmes221/CNNs/raw/master/img/Inceptionv4.png">
    <img src="https://github.com/SherlockHolmes221/CNNs/raw/master/img/Inception-ResNet.png">
</figure>

Give clear empirical evidence that training with residual connections accelerates the training of Inception networks significantly.

Studied how the introduction of residual connections leads to dramatically improved training speed for the Inception architecture.

#### Some doubts on He's point
- Residual connections are inherently necessary for training very deep convolutional models for image recognition.
- Demonstrate that it is not very difficult to train competitive very deep networks without utilizing residual connections. 

#### The evolution of Inception
- GoogLeNet or Inception-v1
- Inception-v2 add BN
- Inception-v3 add factorization ideas 
- Inception-v4 use cheaper Inception blocks combine with resnet

  Each Inception block is followed by filter-expansion layer (1 × 1 convolution without activation) 
  which is used for scaling up the dimensionality 
  of the filter bank before the addition to match the depth of the input.
  
#### Improvement of He's Resnet:Scaling of the Residuals 

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/resnet_scale.png)

- To stabilize the training

#### Experiment details
- RMSProp(momentum=0.9,decay=0.9,$\in$=1.0)
- lr=0.045,decayed every two epochs using an exponential rate of 0.94 

#### Experimental Results

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/inceptionv4_result.png)

- The residual version was training much faster and reached slightly better ﬁnal accuracy than the traditional Inception-v4.
- Although the residual version converges faster, the final accuracy seems to mainly depend on the model size.

#### Some Questions about the point mentioned in the paper:
- Each Inception block is followed by filter-expansion layer (1 × 1 convolution without activation) 
  which is used for scaling up the dimensionality 
  of the filter bank before the addition to match the depth of the input.
- The details about the Scaling of the Residuals.

