# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# Tests for the simple named entity detector.

import unittest

from detect.deep_named_entity_detector import DeepNamedEntityDetector


class TestSimpleNamedEntityDetector(unittest.TestCase):
    def setUp(self):
        """
        Initialise the Simple Named Entity Detector with data for David Cameron.
        """
        self.deepNamedEntityDetector = DeepNamedEntityDetector(3, 1)

    def test_worldCheck1Profiles(self):
        """
        Test the simple named entity detector against World Check 1 profiles
        """
        training_data = [([1.0], "data/profiles/train/davidcameron_1.txt")]
        test_data = [("David Cameron", [1.0], "data/profiles/test/davidcameron_1.txt")]
        self.deepNamedEntityDetector.train("data/temp/deepnet/deepnet.bin", 1, training_data)
        self.deepNamedEntityDetector.test("data/temp/deepnet/deepnet.bin", test_data)


if __name__ == '__main__':
    unittest.main()
