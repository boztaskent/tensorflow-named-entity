# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#

import unittest

from detect.util.wordvec import SimpleWordVec


class TestSimpleWordVec(unittest.TestCase):
    """
    Tests for the simple word vectorizer.
    """

    def setUp(self):
        """
        Test setup
        """
        self.wordVec = SimpleWordVec()

    def test_simpleWordVec(self):
        """
        Test the simple word vectorizer.
        """
        data = "the quick brown fox jumped over the lazy dog"
        result = self.wordVec.vectorize(data.split(), 3)
        expected = [["", "the", "quick"], ["the", "quick", "brown"], ["quick", "brown", "fox"],
                    ["brown", "fox", "jumped"], ["fox", "jumped", "over"], ["jumped", "over", "the"],
                    ["over", "the", "lazy"], ["the", "lazy", "dog"], ["lazy", "dog", ""]]
        self.assertListEqual(expected, result)
