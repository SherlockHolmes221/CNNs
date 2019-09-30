import tensorflow as tf

class SE_Res_50(object):
    def __init__(self):
        self.BN_MOMENTUM = 0.99
        self.BN_EPSILON=9.999999747378752e-06
        self.USE_FUSED_BN = True

    def train(self, input_data):
        conv0 = tf.layers.conv2d(inputs=input_data, filters=64, kernel_size=(7, 7),
                                 strides=(2, 2), padding='same', use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.zeros_initializer())
        conv0_bn_relu_maxpooling = tf.layers.max_pooling2d(tf.nn.relu(
            tf.layers.batch_normalization(conv0, momentum=self.BN_MOMENTUM,epsilon=self.BN_EPSILON, fused=self.USE_FUSED_BN)),
            pool_size=[3,3], strides=[2, 2], padding='same')

        print("conv0_bn_relu:"+str(conv0_bn_relu_maxpooling.shape))#(?, 56, 56, 64)

        input = conv0_bn_relu_maxpooling

        for i in range(3):
            if i == 0:
                output = self.se_bottleneck_block(input, 64, need_reduce=True, is_root=True)
            else:
                output = self.se_bottleneck_block(output, 64, need_reduce=False)

        print("output1"+str(output.shape))#(?, 56, 56, 256)


        for i in range(4):
            if i == 0:
                output = self.se_bottleneck_block(output, 128, need_reduce=True)
            else:
                output = self.se_bottleneck_block(output, 128, need_reduce=False)

        print("output2"+str(output.shape))#(?, 28, 28, 512)

        for i in range(6):
            if i == 0:
                output = self.se_bottleneck_block(output, 256, need_reduce=True)
            else:
                output = self.se_bottleneck_block(output, 256, need_reduce=False)

        print("output3"+str(output.shape))#(?, 14, 14, 1024)

        for i in range(3):
            if i == 0:
                output = self.se_bottleneck_block(output, 512, need_reduce=True)
            else:
                output = self.se_bottleneck_block(output, 512, need_reduce=False)

        print("output3"+str(output.shape))#(?, 7, 7, 2048)

        output = tf.reduce_mean(output, [1, 2], keep_dims=True)
        print('avg'+str(output.shape))#avg(?, 1, 1, 2048)

        output = tf.layers.dense(output, 1000,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True)
        output = tf.nn.softmax(output, name='prob')
        print('output'+str(output.shape))#(?, 1, 1, 1000)
        return output


    def se_bottleneck_block(self, input, filter_num, need_reduce=False, is_root=False, reduced_scale=16):
        residuals = input
        strides = 1

        if need_reduce:
            strides = 1 if is_root else 2
            proj_mapping = tf.layers.conv2d(input, filter_num*4, kernel_size=(1, 1), use_bias=False,
                                            strides=(strides, strides), padding='valid',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            bias_initializer=tf.zeros_initializer())
            residuals = tf.layers.batch_normalization(proj_mapping, momentum=self.BN_MOMENTUM,
                                            epsilon=self.BN_EPSILON, fused=self.USE_FUSED_BN)

        #layer1
        conv1 = tf.layers.conv2d(inputs=input, filters=filter_num, kernel_size=(1, 1),
                                 strides=(strides, strides), padding='same', use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.zeros_initializer())
        conv1_bn = tf.layers.batch_normalization(conv1, momentum=self.BN_MOMENTUM,
                                                   epsilon=self.BN_EPSILON, fused=self.USE_FUSED_BN)
        conv1_bn_relu = tf.nn.relu(conv1_bn)

        #layer2
        conv2 = tf.layers.conv2d(inputs=conv1_bn_relu, filters=filter_num, kernel_size=(3, 3),
                                 strides=(1, 1),padding='same',use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.zeros_initializer())
        conv2_bn = tf.layers.batch_normalization(conv2, momentum=self.BN_MOMENTUM,
                                                   epsilon=self.BN_EPSILON, fused=self.USE_FUSED_BN)
        conv2_bn_relu = tf.nn.relu(conv2_bn)

        #layer3
        conv3 = tf.layers.conv2d(inputs=conv2_bn_relu, filters=filter_num*4, kernel_size=(1, 1),
                                 strides=(1, 1),padding='same',use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.zeros_initializer())
        conv3_bn = tf.layers.batch_normalization(conv3, momentum=self.BN_MOMENTUM,
                                                   epsilon=self.BN_EPSILON, fused=self.USE_FUSED_BN)

        #S
        pooled_inputs = tf.reduce_mean(conv3_bn, [1, 2], keep_dims=True)
        print("pooled_inputs:"+str(pooled_inputs.shape))#(?, 1, 1, 256)

        #E
        down_inputs = tf.layers.conv2d(pooled_inputs, (filter_num * 4) // reduced_scale,
                                       kernel_size=(1, 1), use_bias=True,strides=(1, 1),
                                       padding='valid',  activation=None,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.zeros_initializer())
        down_inputs_relu = tf.nn.relu(down_inputs)
        print("down_inputs:"+str(down_inputs.shape))#(?, 1, 1, 16)

        up_inputs = tf.layers.conv2d(down_inputs_relu, filter_num * 4, kernel_size=(1, 1),
                                     use_bias=True, strides=(1, 1),
                                     padding='valid', activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.zeros_initializer())
        prob_outputs = tf.nn.sigmoid(up_inputs)
        print("up_inputs:"+str(up_inputs.shape))#(?, 1, 1, 256)

        rescaled_feat = tf.multiply(prob_outputs, conv3_bn)

        pre_act = tf.add(residuals, rescaled_feat)
        return tf.nn.relu(pre_act)


if __name__ == '__main__':
    input_data = tf.placeholder(dtype=tf.float32, name='input_data', shape=[None, 224, 224, 3])
    SE_Res_50().train(input_data)




