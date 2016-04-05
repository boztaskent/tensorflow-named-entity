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

        # x_text = [s.split(" ") for s in x_text]

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

    def pad_sentences(self, sentences, padding_word="<PAD/>"):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        sequence_length = max(len(x) for x in sentences)
        padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = sequence_length - len(sentence)
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
        Maps sentencs and labels to vectors based on a vocabulary.
        """
        x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
        y = np.array(labels)
        return [x, y]


    def load_data(self, training_data, max_words):
        """
        Loads and preprocessed data for the MR dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        sentences, labels = self.load_data_and_labels(training_data, max_words)
        sentences_padded = self.pad_sentences(sentences)
        vocabulary, vocabulary_inv = self.build_vocab(sentences_padded)
        x, y = self.build_input_data(sentences_padded, labels, vocabulary)
        return [x, y, vocabulary, vocabulary_inv]


    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]