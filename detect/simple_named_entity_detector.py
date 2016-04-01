# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# The implementation of a simple named entity detector.

from named_entity_detector import NamedEntityDetector


class SimpleNamedEntityDetector(NamedEntityDetector):
    """
      A simple named entity detector.
    """

    def __init__(self, network_name, positive_entities, negative_entities):
        """
        Initialise the instance.

        :param network_name the neural network name
        :param positive_entities the list of positive entities we are detecting
        :param negative_entities the list of negative entities used for training
        """
        NamedEntityDetector.__init__(self, network_name, positive_entities, negative_entities)

    def train(self):
        """
        Train the neural network
        """
        pass
