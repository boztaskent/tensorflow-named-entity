# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from detect.util.wordvec import SimpleWordVec


class DataSet(object):
    """
    This class defines a dataset, it provides an iterator that returns tagged word vectors.
    """

    def __init__(self, window_size, training_data):
        """
        Read all training data and convert to word vectors for later iteration
        """
        self.labels = dict()
        self.labels[""] = 0
        word_vec = SimpleWordVec()
        self.window_size = window_size
        self.training_data = training_data
        self.word_vectors = dict()
        for i in range(len(training_data)):
            words = self.read_words(training_data[i][1])
            mapped_words = map(self.get_label, words)
            self.word_vectors[i] = word_vec.vectorize(mapped_words, self.window_size, 0)
        self.data_idx = 0
        self.vector_idx = dict()
        self.reset()

    def reset(self):
        """
        Reset the data index position for the get data iterator
        """
        self.data_idx = 0
        for i in range(len(self.training_data)):
            self.vector_idx[i] = 0

    def get_data(self):
        """
        Fetch the next data vector from the input
        """
        word_vectors = self.word_vectors[self.data_idx]
        result = [self.training_data[self.data_idx][0], word_vectors[self.vector_idx[self.data_idx]]]
        self.vector_idx[self.data_idx] += 1
        if self.vector_idx[self.data_idx] >= len(word_vectors):
            self.vector_idx[self.data_idx] = 0
        self.data_idx += 1
        if self.data_idx >= len(self.training_data):
            self.data_idx = 0
        return result

    def read_words(self, filename):
        """
        Read a file from input and return an array of words.

        :param filename the filename to be read, assumed to be relative to this folder
        """
        with open(os.path.dirname(__file__) + "/../" + filename, "r") as myfile:
            return myfile.read().replace("\n", " ").split()

    def get_label(self, word):
        """
        Get a label from the dictionary for a specific word. If the word is not in the
        dictionary then add it and return the label that is generated.
        """
        if word not in self.labels.keys():
            self.labels[word] = len(self.labels) * 0.000001
        return self.labels[word]
