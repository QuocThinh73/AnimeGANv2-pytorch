from functools import reduce
import tensorflow as tf
import numpy as np
import time
import sys

VGG_MEAN = [103.939, 116.779, 123.68]  # B, G, R


class VGG16:
    def __init__(self, vgg16_npy_path='vgg16_weight/vgg16.npy'):
        if vgg16_npy_path:
            self.data_dict = np.load(
                vgg16_npy_path, encoding='latin1', allow_pickle=True).item()
            print("npy file loaded ------- ", vgg16_npy_path)
        else:
            self.data_dict = None
            print("npy file load error!")
            sys.exit(1)

    def build(self, rgb, include_fc=False):
        """
        input: RGB in [-1, 1] which will be converted into BGR mean
        """
        start_time = time.time()
        # [-1, 1] -> [0, 1] -> [0, 255]
        rgb_scaled = ((rgb + 1.0) / 2.0) * 255.0
        red, green, blue = tf.split(
            axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(
            axis=3, values=[blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]])

        # VGG16 backbone (13 conv)
        self.conv1_1 = self.convolution_layer(bgr, "conv1_1")
        self.conv1_2 = self.convolution_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, "pool1")

        self.conv2_1 = self.convolution_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.convolution_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, "pool2")

        self.conv3_1 = self.convolution_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.convolution_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.convolution_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, "pool3")

        self.conv4_1 = self.convolution_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.convolution_layer(self.conv4_1, "conv4_2")

        self.conv4_3_no_activation = self.no_activation_convolution_layer(
            self.conv4_2, "conv4_3")

        self.conv4_3 = self.convolution_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, "pool4")

        self.conv5_1 = self.convolution_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.convolution_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.convolution_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, "pool5")

        if include_fc:
            self.fc6 = self.fc_layer(self.pool5, "fc6")
            self.relu6 = tf.nn.relu(self.fc6)
            self.fc7 = self.fc_layer(self.relu6, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)
            self.fc8 = self.fc_layer(self.relu7, "fc8")
            self.prob = tf.nn.softmax(self.fc8, name="prob")
            self.data_dict = None

        print(f"build model VGG16 finshed: {time.time() - start_time}")

    def average_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def convolution_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name):
            x = self.no_activation_convolution_layer(bottom, name)
            return tf.nn.relu(x)

    def no_activation_convolution_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name):
            filter = self.get_convolutional_filter(name)
            convolution = tf.nn.conv2d(
                bottom, filter, [1, 1, 1, 1], padding="SAME")
            biases = self.get_bias(name)
            x = tf.nn.bias_add(convolution, biases)
            return x

    def fc_layer(self, bottom, name):
        with tf.compat.v1.get_variable(name):
            shape = bottom.get_shape().as_list()
            dim = reduce(lambda x, y: x * y, shape[1:], 1)
            x = tf.reshape(bottom, [-1, dim])
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_convolutional_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
