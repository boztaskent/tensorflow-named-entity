#!/usr/bin/python
#
# The implementation of a simple named entity detector.

from named_entity_detector import NamedEntityDetector


class SimpleNamedEntityDetector(NamedEntityDetector):
    """
      Train the simple named entity detector.
    """

    def __init__(self, dataFile):
        """
        Initialise the instance.
        """
        NamedEntityDetector.__init__(self, dataFile)
