# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# The implementation of a simple named entity detector.

import os

import tensorflow as tf

from named_entity_detector import NamedEntityDetector
from util.wordvec import SimpleWordVec


class DeepNamedEntityDetector(NamedEntityDetector):
    """
      A deep neural network named entity detector.
    """

    def __init__(self, window_size, num_entities, multi_factor=10):
        """
        Initialise the instance.
        """
        NamedEntityDetector.__init__(self)

        self.window_size = window_size
        self.num_entities = num_entities
        self.mult_factor = multi_factor
        self.x = tf.placeholder(tf.float32, [None, window_size * self.mult_factor])
        self.W = tf.Variable(tf.zeros([window_size * self.mult_factor, num_entities]), name="weights")
        self.b = tf.Variable(tf.zeros([num_entities]), name="bias")
        self.y_ = tf.placeholder(tf.float32, [None, num_entities])

        self.W_conv1 = self.weight_variable([1, 1, 1, 1])
        self.b_conv1 = self.bias_variable([1])
        self.x_image = tf.reshape(self.x, [-1, 1, 1, 1])
        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        self.W_conv2 = self.weight_variable([1, 1, 1, 1])
        self.b_conv2 = self.bias_variable([1])
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        self.W_fc1 = self.weight_variable([3, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 3])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc2 = self.weight_variable([1024, num_entities])
        self.b_fc2 = self.bias_variable([num_entities])
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

    def train(self, network_name, outer_steps, inner_steps, training_data):
        """
        Train the neural network
        """
        word_vec = SimpleWordVec()
        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        for x in range(outer_steps):
            for j in range(len(training_data)):
                words = self.read_words(training_data[j][1])
                mapped_words = map(self.get_label, words)
                word_vectors = word_vec.vectorize(mapped_words, self.window_size, 0)
                for k in range(len(word_vectors)):
                    for i in range(inner_steps):
                        batch_xs = [word_vectors[k] * self.mult_factor]
                        batch_ys = [training_data[j][0]]
                        sess.run(train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.5})
            output_file = os.path.dirname(__file__) + "/" + network_name
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        saver.save(sess, output_file)
        sess.close()

    def run(self, network_name, test_data):
        """
        Restore the network with the specified name and then run the test data against it.
        """
        word_vec = SimpleWordVec()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.dirname(__file__) + "/" + network_name)
        for i in range(len(test_data)):
            words = self.read_words(test_data[i])
            mapped_words = map(self.get_label, words)
            word_vectors = word_vec.vectorize(mapped_words, self.window_size, 0)

            calc = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)
            score = [0.0] * self.num_entities
            for j in range(len(word_vectors)):
                batch_xs = [word_vectors[j] * self.mult_factor]
                result = sess.run(calc, feed_dict={self.x: batch_xs, self.keep_prob: 0.5})
                for k in range(self.num_entities):
                    score[k] += result[0][k]
            print "Raw score: ", score
            print "Softmax score: ", self.softmax(score)

        sess.close()

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 1, 1, 1],
                              strides=[1, 1, 1, 1], padding='SAME')
