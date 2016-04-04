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
        self.simpleNamedEntityDetector = SimpleNamedEntityDetector(5, 6, 200)

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

        self.simpleNamedEntityDetector.train("data/temp/simplenet/simplenet.bin", 5, 5, training_data)

        test_data = [
            "data/profiles/test/davidcameron_1.txt",
            "data/profiles/test/davidcameron_2.txt"
        ]
        self.simpleNamedEntityDetector.run("data/temp/simplenet/simplenet.bin", test_data)


if __name__ == '__main__':
    unittest.main()
