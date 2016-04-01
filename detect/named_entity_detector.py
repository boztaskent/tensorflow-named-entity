# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# The base class for all named entity detectors.

import re
from HTMLParser import HTMLParser
from util.word2vec import Word2Vec
import requests
import yaml


class HTMLStripper(HTMLParser):
    """
    Strip HTML tags and return only the raw data within the HTML content.
    """

    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


class NamedEntityDetector:
    """
      The base class for all named entity detectors.
    """

    def __init__(self, network_name, positive_entities, negative_entities):
        """
        Initialise the new instance
        """
        self.network_name = network_name
        self.positive_entities = positive_entities
        self.negative_entities = negative_entities

    def readUrl(self, url):
        """
        Read the contents of a URL and return as a string
        """
        return requests.get(url).content

    def stripHtml(self, data):
        """
        Strip the HTML from a web page of data and return the content.
        """
        data = data.replace("\n", " ")
        data = re.sub("<script.*?script>", "", data, 0, re.MULTILINE)
        data = re.sub("<style.*?style>", "", data, 0, re.MULTILINE)
        htmlParser = HTMLStripper()
        htmlParser.feed(data)
        return htmlParser.get_data()

    def tokensize(self, data):
        """
        Tokensize the input string into an array of words.
        """
        return data.split()

    def train(self):
        """
        Train the neural network.
        """
        raise Exception("Must be implemented by sub-class")

    def detect(self, url):
        """
        Detect the named entities contained in the given url
        """
        raise Exception("Must be implemented by sub-class")
