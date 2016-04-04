# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# The base class for all named entity detectors.

import os


class NamedEntityDetector:
    """
      The base class for all named entity detectors.
    """

    def __init__(self):
        """
        Initialise the new instance
        """
        self.labels = dict()
        self.labels[""] = 0

    def read_words(self, filename):
        """
        Read a file from input and return an array of words.

        :param filename the filename to be read, assumed to be relative to this folder
        """
        with open(os.path.dirname(__file__) + "/" + filename, "r") as myfile:
            return myfile.read().replace("\n", " ").split()

    def get_label(self, word):
        """
        Get a label from the dictionary for a specific word. If the word is not in the
        dictionary then add it and return the label that is generated.
        """
        if word not in self.labels.keys():
            self.labels[word] = len(self.labels)
        return self.labels[word]
