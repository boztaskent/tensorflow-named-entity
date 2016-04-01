# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# The implementation of a simple named entity detector.

from named_entity_detector import NamedEntityDetector


class SimpleNamedEntityDetector(NamedEntityDetector):
    """
      A simple named entity detector.
    """

    def __init__(self, network_name, entity, urls):
        """
        Initialise the instance.
        """
        NamedEntityDetector.__init__(self, network_name, entity, urls)

    def train(self):
        """
        Train the neural network
        """
        pass
