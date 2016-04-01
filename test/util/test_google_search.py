# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#

import unittest

from detect.util.google_search import GoogleSearch


class TestGoogleSearch(unittest.TestCase):
    """
    Tests for the Google search utility class.
    """

    def setUp(self):
        """
        Test setup.
        """
        self.googleSearch = GoogleSearch()

    def test_search(self):
        """
        Test Google search against a fairly popular search term and ensure
        that it returns at least the number of results requested.
        """
        results = self.googleSearch.search("David Cameron", 20)
        self.assertEqual(20, len(results))
