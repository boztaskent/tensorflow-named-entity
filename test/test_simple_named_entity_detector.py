# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# Tests for the simple named entity detector.

import unittest

from detect.simple_named_entity_detector import SimpleNamedEntityDetector


class TestSimpleNamedEntityDetector(unittest.TestCase):
    def setUp(self):
        """
        Initialise the Simple Named Entity Detector with data for David Cameron.
        """
        self.simpleNamedEntityDetector = SimpleNamedEntityDetector("test", [], [])

if __name__ == '__main__':
    unittest.main()
