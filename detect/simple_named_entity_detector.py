# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# The implementation of a simple named entity detector.

import os

import tensorflow as tf

import numpy as np

from named_entity_detector import NamedEntityDetector
from util.data_helpers import DataHelper


class SimpleNamedEntityDetector(NamedEntityDetector):
    """
      A simple named entity detector.
    """

    def __init__(self, input_size, num_entities):
        """
        Initialise the instance.
        """
        NamedEntityDetector.__init__(self)

        self.input_size = input_size
        self.num_entities = num_entities
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.W = tf.Variable(tf.zeros([self.input_size, num_entities]), name="weights")
        self.b = tf.Variable(tf.zeros([num_entities]), name="bias")
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        self.y_ = tf.placeholder(tf.float32, [None, num_entities])

    def train(self, network_name, num_epochs, batch_size, training_data):
        """
        Train the neural network
        """
        # cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
        loss = tf.reduce_mean(tf.square(self.y_ - self.y))
        # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        data_helper = DataHelper()
        x, y, vocabulary, _ = data_helper.load_data(training_data, self.input_size)
        self.vocabulary = vocabulary

        batches = data_helper.batch_iter(x, y, batch_size, num_epochs, shuffle=True)
        for batch_xs, batch_ys in batches:
            sess.run(train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
            # print("Loss: ", sess.run(loss, feed_dict={self.x: batch_xs, self.y_: batch_ys}))

        output_file = os.path.dirname(__file__) + "/" + network_name
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        saver.save(sess, output_file)
        sess.close()

    def run(self, network_name, test_data):
        """
        Restore the network with the specified name and then run the test data against it.
        """
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.dirname(__file__) + "/" + network_name)

        data_helper = DataHelper()

        # for i in range(len(test_data)):
        #     x, y = data_helper.load_data_using_voc([test_data[i]], self.input_size, self.vocabulary)
        #     calc = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        #     # score = [0.0] * self.num_entities
        #     result = sess.run(calc, feed_dict={self.x: x})

        score = [0.0] * self.num_entities
        x, y = data_helper.load_data_using_voc([test_data[0]], self.input_size, self.vocabulary)
        calc = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        result = sess.run(calc, feed_dict={self.x: x})
        for j in range(len(result)):
            max_index = np.argmax(result[j])
            score[max_index] = score[max_index] + 1

        print("x-size: ", len(x))
        print(score)
            #     for k in range(self.num_entities):
            #         score[k] += result[0][k]
            # print "Raw score: ", score
            # print "Softmax score: ", self.softmax(score)
        sess.close()
