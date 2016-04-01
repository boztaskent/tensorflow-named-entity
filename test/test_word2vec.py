# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# Tests for basic word2vec mapping and visualisation

import os
import unittest
from detect.word2vec import run_word2vec


class TestWord2Vec(unittest.TestCase):
    """
    Tests for the basic word2vec application and visualisation.
    """

    def test_word2vec(self):
        run_word2vec()
