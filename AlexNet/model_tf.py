import tensorflow as tf
import numpy as np


class Alexnet():
    def __init__(self):
        print('alex_tf')

    def train(self, x_train, y_train):
        conv1 = self.conv_relu(x_train, 11, 11, 96, 4, 4, 'conv1', 'VALID')
        lrn1 = self.lrn(conv1, 2, 2e-05, 0.75, 'lrn1')
        pool1 = self.max_pooling(lrn1, 3, 3, 2, 2, padding='VALID', name='pool1')

        conv2 = self.conv_relu(pool1, 5, 5, 256, 1, 1, name='conv2', groups=2)
        lrn2 = self.lrn(conv2, 2, 2e-05, 0.75, 'lrn2')
        pool2 = self.max_pooling(lrn2, 3, 3, 2, 2, padding='VALID', name='pool2')

        conv3 = self.conv_relu(pool2, 3, 3, 384, 1, 1, name='conv3')

        conv4 = self.conv_relu(conv3, 3, 3, 384, 1, 1, name='conv4', groups=2)

        conv5 = self.conv_relu(conv4, 3, 3, 256, 1, 1, name='conv5', groups=2)
        pool5 = self.max_pooling(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = self.fc(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = self.drop_out(fc6, 0.5)

        fc7 = self.fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = self.drop_out(fc7, 0.5)

        self.fc8 = self.fc(dropout7, 4096, 1000, relu=False, name='fc8')


    def conv_relu(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
        input_channels = int(x.get_shape()[-1])

        # Create lambda function for the convolution
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height,
                                                        filter_width,
                                                        input_channels/groups,
                                                        num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

            if groups == 1:
                conv = convolve(x, weights)
            else:
                # Split input and weights and convolve them separately
                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups,value=weights)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
                # Concat the convolved output together again
                conv = tf.concat(axis=3, values=output_groups)
            # Add biases
            bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
            # Apply relu function
            relu = tf.nn.relu(bias, name=scope.name)
            return relu


    def fc(self, x,  input_num, output_num, name, relu=True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weight', shape=[input_num, output_num], trainable=True)
            biases = tf.get_variable('biases', [output_num], trainable=True)
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        if relu:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

    def max_pooling(self, x , filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool2d(x, ksize=[1, filter_height, filter_width, 1],
                                strides=[1, stride_y, stride_x, 1],
                                padding=padding, name=name)

    def drop_out(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)

    def lrn(x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius,
                                                  alpha=alpha, beta=beta,
                                                  bias=bias, name=name)


if __name__ == '__main__':
    net = Alexnet()
