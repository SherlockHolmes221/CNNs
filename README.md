# 卷积神经网络总结

### AlexNet
###### Group convolution
- 分组卷积，训练AlexNet时卷积操作不能全部放在同一个GPU处理，因此作者把feature maps分给多个GPU分别进行处理，最后把多个GPU的结果进行融合。
- 分组卷积最后每一组输出的feature maps应该是以concatenate的方式组合，而不是element-wise add，所以每组输出的channel是 input channels / #groups，这样参数量就大大减少了。

### VGG 
###### 3*3卷积核
- 大的卷积核会导致计算量的暴增，不利于模型深度的增加，计算性能也会降低。
- 2个3×3卷积核的组合比1个5×5卷积核的效果更佳，同时参数量（3×3×2+1 VS 5×5×1+1）被降低

### Inception
###### 同一层feature map可以分别使用多个不同尺寸的卷积核
- 获得不同尺度的特征，再把这些特征结合起来，得到的特征往往比使用单一卷积核的要好
- 缺点:参数量比单个卷积核要多很多，如此庞大的计算量会使得模型效率低下
###### 1×1的卷积核
- 解决引入多个尺寸的卷积核，会带来大量的额外的参数的问题

### ResNet
- 当层数加深时，网络的表现越来越差，很大程度上的原因是因为当层数加深时，梯度消散得越来越严重，以至于反向传播很难训练到浅层的网络。
- 为了解决这个问题，“残差网络”使得梯度更容易地流动到浅层的网络当中去，

### Xception
###### DepthWise
- 卷积操作时须不需要同时考虑通道和区域
- 我们首先对每一个通道进行各自的卷积操作，有多少个通道就有多少个过滤器。得到新的通道feature maps之后，这时再对这批新的通道feature maps进行标准的1×1跨通道卷积操作。这种操作被称为 “DepthWise convolution” ，缩写“DW”

###  ShuffleNet

### SEnet


###  Dilated convolution

### Deformable convolution
