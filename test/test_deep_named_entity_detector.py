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
        self.deepNamedEntityDetector = DeepNamedEntityDetector(3, 6)

    def test_worldCheck1Profiles(self):
        """
        Test the simple named entity detector against World Check 1 profiles
        """
        training_data = [
            ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "data/profiles/train/davidcameron_1.txt"),
            ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "data/profiles/train/davidcameron_2.txt"),
            ([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "data/profiles/train/davidcameron_3.txt"),
            ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "data/profiles/train/davidcameron_4.txt"),
            ([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "data/profiles/train/davidcameron_5.txt"),
            ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "data/profiles/train/davidcameron_6.txt"),
        ]
        self.deepNamedEntityDetector.train("data/temp/deepnet/deepnet.bin", 2, training_data)

        test_data = ["data/profiles/test/davidcameron_1.txt"]
        self.deepNamedEntityDetector.run("data/temp/deepnet/deepnet.bin", test_data)


if __name__ == '__main__':
    unittest.main()
