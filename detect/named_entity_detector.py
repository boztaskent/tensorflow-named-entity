# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# The base class for all named entity detectors.


class NamedEntityDetector:
    """
      The base class for all named entity detectors.
    """

    def __init__(self, network_name, entity, urls):
        """
        Initialise the new instance

        :param network_name the neural network name
        :param entity a vector representing the entity we wish to learn
        :param urls urls that positively classify the entity
        """
        self.network_name = network_name
        self.entity = entity
        self.urls = urls

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
