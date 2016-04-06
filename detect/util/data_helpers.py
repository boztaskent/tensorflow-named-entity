import os
import numpy as np
import re
import itertools
import nltk
from collections import Counter


class DataHelper(object):

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def load_data_and_labels(self, training_data, max_words):
        x_text = []
        y = []
        for i in range(len(training_data)):
            label = training_data[i][0]
            filename = training_data[i][1]
            sentences = self.get_sentences(filename, max_words)
            for _ in sentences:
                y.append(label)
            for sent in sentences:
                x_text.append(sent)

        return [x_text, y]

    def get_sentences(self, filename, max_words):
        with open(os.path.dirname(__file__) + "/../" + filename, "r") as my_file:
            contents = my_file.read()
            sentences = nltk.sent_tokenize(contents)
            sentences = [self.clean_str(sent) for sent in sentences]
            sentences = [nltk.word_tokenize(sent) for sent in sentences]
            valid_sentences = []
            for sent in sentences:
                if len(sent) > max_words:
                    for i in xrange(0, len(sent), max_words):
                        valid_sentences.append(sent[i:i+max_words])
                else:
                    valid_sentences.append(sent)
            return valid_sentences

    def pad_sentences(self, sentences, max_words):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        padding_word = '<NONE/>'
        padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = max_words - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        return padded_sentences


    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]


    def build_input_data(self, sentences, labels, vocabulary):
        """
        Maps sentences and labels to vectors based on a vocabulary.
        """
        x = np.array([[self.getWordValueFromVocabulary(word, vocabulary) for word in sentence] for sentence in sentences])
        y = np.array(labels)
        return [x, y]

    def getWordValueFromVocabulary(self, word, vocabulary):
        if word in vocabulary:
            return vocabulary[word] * 1.0 / len(vocabulary)
        else:
            return vocabulary['<NONE/>'] * 1.0 / len(vocabulary)

    def load_data(self, data, max_words):
        # Load and preprocess data
        sentences, labels = self.load_data_and_labels(data, max_words)
        sentences_padded = self.pad_sentences(sentences, max_words)
        vocabulary, vocabulary_inv = self.build_vocab(sentences_padded)
        x, y = self.build_input_data(sentences_padded, labels, vocabulary)
        return [x, y, vocabulary, vocabulary_inv]


    def load_data_using_voc(self, data, max_words, vocabulary):
        # Load and preprocess data
        sentences, labels = self.load_data_and_labels(data, max_words)
        sentences_padded = self.pad_sentences(sentences, max_words)
        x, y = self.build_input_data(sentences_padded, labels, vocabulary)
        return [x, y]


    def batch_iter(self, data_x, data_y, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        data_size = len(data_x)
        num_batches_per_epoch = int(len(data_x)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data_x = data_x[shuffle_indices]
                shuffled_data_y = data_y[shuffle_indices]
            else:
                shuffled_data_x = data_x
                shuffled_data_y = data_y
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data_x[start_index:end_index], shuffled_data_y[start_index:end_index]