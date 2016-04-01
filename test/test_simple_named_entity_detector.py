# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# Tests for the simple named entity detector.

import os
import unittest

from detect.simple_named_entity_detector import SimpleNamedEntityDetector


class TestSimpleNamedEntityDetector(unittest.TestCase):
    def setUp(self):
        """
        Initialise the Simple Named Entity Detector with data for David Cameron.
        """
        self.simpleNamedEntityDetector = SimpleNamedEntityDetector(os.path.dirname(__file__) + "/data/davidcameron.yml")

    def test_readUrl(self):
        """
        This test is a bit fragile as it checks the number of bytes read, unfortunately nothing on the
        Internet can be guaranteed to remain the same.
        """
        data = self.simpleNamedEntityDetector.readUrl("http://www.arshadmahmood.me")
        self.assertEqual(44647, len(data))

    def test_stripHtml(self):
        """
        Again a fragile test as it relies on the website not changing, but this is the easiest way give
        a complex HTML file for stripping.
        """
        data = self.simpleNamedEntityDetector.readUrl("http://www.arshadmahmood.me")
        stripped_data = self.simpleNamedEntityDetector.stripHtml(data)
        self.assertEqual(651, len(stripped_data))


if __name__ == '__main__':
    unittest.main()
