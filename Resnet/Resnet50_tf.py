import tensorflow as tf
import tensorflow
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_50


class Res50(object):
    def __init__(self):
        self.CONV_WEIGHT_STDDEV = 0.1
        self.CONV_WEIGHT_DECAY = 0.00004
        self.slim = tf.contrib.slim

    def train(self, x_input):
        conv_0 = self.conv(x_input, 7, 64, 2, 2, name='conv_0')
        conv_0_bn_relu = self.bn_relu(conv_0)
        print(conv_0_bn_relu.shape)#(1, 112, 112, 64)

        conv_0_bn_relu_maxpooling = self.max_pooling(conv_0_bn_relu, 3, 2)
        x1 = conv_0_bn_relu_maxpooling
        print(x1.shape) #(1, 56, 56, 64)

        for i in range(3):
            conv1_1 = self.conv(x1, 1, 64, 1, 1, name="conv1_1_%d" % i)
            conv1_1_bn_relu = self.bn_relu(conv1_1)

            conv1_2 = self.conv(conv1_1_bn_relu, 3, 64, 1, 1, name="conv1_2_%d" % i)
            conv1_2_bn_relu = self.bn_relu(conv1_2)

            conv1_3 = self.conv(conv1_2_bn_relu, 1, 256, 1, 1, name="conv1_3_%d" % i)
            print(conv1_3.shape)#(1, 56, 56, 256)

            print(x1.shape)#(1, 56, 56, 64)
            x1 = self.shortcut(x1, conv1_3, name='shortcut_1_%d' % i)
            print(x1.shape)#(1, 56, 56, 256)

        x1 = self.bn_relu(x1)
        print("end1"+str(x1.shape))#(1, 56, 56, 256)

        for i in range(4):
            if i == 0:
                conv2_1 = self.conv(x1, 1, 128, 2, 2, name="conv2_1_%d" % i)
            else:
                conv2_1 = self.conv(x1, 1, 128, 1, 1, name="conv2_1_%d" % i)

            conv2_1_bn_relu = self.bn_relu(conv2_1)

            conv2_2 = self.conv(conv2_1_bn_relu, 3, 128, 1, 1, name="conv2_2_%d" % i)
            conv2_2_bn_relu = self.bn_relu(conv2_2)

            conv2_3 = self.conv(conv2_2_bn_relu, 1, 512, 1, 1, name="conv2_3_%d" % i)
            print(conv2_3.shape)#(1, 28, 28, 512)

            print(x1.shape)#(1, 56, 56, 256)
            x1 = self.shortcut(x1, conv2_3, name='shortcut_2_%d' % i)
            print(x1.shape)#(1, 28, 28, 512)

        x1 = self.bn_relu(x1)
        print("end1"+str(x1.shape))

        for i in range(6):
            if i == 0:
                conv3_1 = self.conv(x1, 1, 256, 2, 2, name="conv3_1_%d" % i)
            else:
                conv3_1 = self.conv(x1, 1, 256, 1, 1, name="conv3_1_%d" % i)

            conv3_1_bn_relu = self.bn_relu(conv3_1)

            conv3_2 = self.conv(conv3_1_bn_relu, 3, 256, 1, 1, name="conv3_2_%d" % i)
            conv3_2_bn_relu = self.bn_relu(conv3_2)

            conv3_3 = self.conv(conv3_2_bn_relu, 1, 1024, 1, 1, name="conv3_3_%d" % i)
            print(conv3_3.shape)

            print(x1.shape)
            x1 = self.shortcut(x1, conv3_3, name='shortcut_3_%d' % i)
            print(x1.shape)

        x1 = self.bn_relu(x1)
        print("end1"+str(x1.shape))

        for i in range(6):
            if i == 0:
                conv4_1 = self.conv(x1, 1, 512, 2, 2, name="conv4_1_%d" % i)
            else:
                conv4_1 = self.conv(x1, 1, 512, 1, 1, name="conv4_1_%d" % i)

            conv4_1_bn_relu = self.bn_relu(conv4_1)

            conv4_2 = self.conv(conv4_1_bn_relu, 3, 512, 1, 1, name="conv4_2_%d" % i)
            conv4_2_bn_relu = self.bn_relu(conv4_2)

            conv4_3 = self.conv(conv4_2_bn_relu, 1, 2048, 1, 1, name="conv4_3_%d" % i)
            print(conv4_3.shape)

            print(x1.shape)
            x1 = self.shortcut(x1, conv4_3, name='shortcut_4_%d' % i)
            print(x1.shape)

        x1 = self.bn_relu(x1)
        print("end1"+str(x1.shape))

        x1 = self.avg_pooling(x1, 7, 1)
        x1 = self.fc(x1, 2048, 1000, name='fc', softmax=True)

        return x1

    def conv(self, input, filter_size, filter_num, stride_h, stride_w, name, padding='SAME',bias=True):
        input_channels = int(input.get_shape()[-1])
        initializer = tf.truncated_normal_initializer(stddev=self.CONV_WEIGHT_STDDEV)

        regularizer = tf.contrib.layers.l2_regularizer(self.CONV_WEIGHT_DECAY)
        filter =tf.get_variable(name+'_weights', shape=[filter_size,
                                                    filter_size,
                                                    input_channels,
                                                    filter_num],
                                initializer=initializer,
                                dtype='float',
                                regularizer=regularizer,
                                trainable=True)

        convlution = tf.nn.conv2d(input=input,
                                  filter=filter,
                                  strides=[1, stride_h, stride_w, 1],
                                  padding=padding,
                                  name=name)
        if bias:
            biases = tf.get_variable(name+'_biases', shape=[filter_num])
            convlution = tf.reshape(tf.nn.bias_add(convlution, biases), tf.shape(convlution))

        return convlution

    def max_pooling(self, input, filter_size, stride, padding='SAME'):
        return tf.nn.max_pool2d(input,
                              ksize=[1, filter_size, filter_size, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding)

    def avg_pooling(self, x, filter, stride, padding='SAME'):
        return tf.nn.avg_pool2d(x, ksize=[1, filter, filter, 1],
                                strides=[1, stride, stride, 1],
                                padding=padding)

    def bn_relu(self, input):
        conv = tf.layers.batch_normalization(input, beta_initializer=tf.zeros_initializer(),
                                             gamma_initializer=tf.ones_initializer(),
                                             moving_mean_initializer=tf.zeros_initializer(),
                                             moving_variance_initializer=tf.ones_initializer())
        conv = tf.nn.relu(conv)
        return conv

    def fc(self, x,  input_num, output_num, name, softmax=True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable(name+'_weight', shape=[input_num, output_num], trainable=True)
            biases = tf.get_variable(name+'_biases', [output_num], trainable=True)
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        if softmax:
            softmax = tf.nn.softmax(act)
            return softmax
        else:
            return act

    def shortcut(self, input, residual, name):
        """Adds a shortcut between input and residual block and merges them with "sum"
        """
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        residual_shape = residual.shape.as_list()
        input_shape = input.shape.as_list()

        stride_width = int(round(input_shape[2] / residual_shape[2]))
        stride_height = int(round(input_shape[1]/ residual_shape[1]))

        equal_channels = input_shape[-1] == residual_shape[-1]

        shortcut = input

        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = self.conv(input, 1, residual_shape[-1], stride_height,
                                 stride_width, name=name, padding="VALID")
        return shortcut + residual


if __name__ == '__main__':
    input_data = tf.placeholder(dtype=tf.float32, name='input_data', shape=[1, 224, 224, 3])
    Res50().train(input_data)
