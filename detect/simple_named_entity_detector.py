# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# The implementation of a simple named entity detector.

import os
import math
import tensorflow as tf
from named_entity_detector import NamedEntityDetector
from util.wordvec import SimpleWordVec


class SimpleNamedEntityDetector(NamedEntityDetector):
    """
      A simple named entity detector.
    """

    def __init__(self, window_size, num_entities):
        """
        Initialise the instance.
        """
        NamedEntityDetector.__init__(self)

        self.window_size = window_size
        self.num_entities = num_entities
        self.x = tf.placeholder(tf.float32, [None, window_size])
        self.W = tf.Variable(tf.zeros([window_size, num_entities]), name="weights")
        self.b = tf.Variable(tf.zeros([num_entities]), name="bias")
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        self.y_ = tf.placeholder(tf.float32, [None, num_entities])

    def train(self, network_name, num_steps, training_data):
        """
        Train the neural network
        """
        word_vec = SimpleWordVec()
        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        for i in range(num_steps):
            for j in range(len(training_data)):
                words = self.read_words(training_data[j][1])
                mapped_words = map(self.get_label, words)
                word_vectors = word_vec.vectorize(mapped_words, self.window_size, 0)
                for k in range(len(word_vectors)):
                    batch_xs = [word_vectors[k]]
                    batch_ys = [training_data[j][0]]
                    sess.run(train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
        output_file = os.path.dirname(__file__) + "/" + network_name
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        saver.save(sess, output_file)
        sess.close()

    def test(self, network_name, test_data):
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

            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            score = 0.0
            for j in range(len(word_vectors)):
                batch_xs = [word_vectors[j]]
                score += sess.run(accuracy, feed_dict={self.x: batch_xs})
            print score/len(word_vectors)
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

            calc = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
            score = [0.0] * self.num_entities
            for j in range(len(word_vectors)):
                batch_xs = [word_vectors[j]]
                result = sess.run(calc, feed_dict={self.x: batch_xs})
                for k in range(self.num_entities):
                    score[k] += result[0][k]
            print "Raw score: ", score
            print "Softmax score: ", self.softmax(score)
        sess.close()
