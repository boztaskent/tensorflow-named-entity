# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


class WordVec(object):
    """
    The base class for all word vector generators, essentially these take
    an input string and convert it to a set of vectors of a given size.
    """

    def vectorize(self, data, window_size):
        """
        Generate vectors from the input data.
        :param data the string to be vectorized
        :param window_size the length of the vectors
        """
        raise Exception("Must be implemented by sub-class")


# ==============================================================================
# SimpleWordVec: A simple implementation of word vectorisation.
# ==============================================================================

class SimpleWordVec(WordVec):
    """
    A simple implementation of word vectorisation with random labelling.
    """

    def vectorize(self, data, window_size, empty_value=""):
        """
        Vectorize the word.
        """
        result = []
        for i in range(0, len(data)):
            idx1 = i - (window_size / 2)
            idx2 = i + (window_size / 2) + 1
            element = data[max(idx1, 0):min(idx2, len(data))]
            if idx1 < 0:
                element = ([empty_value] * abs(idx1)) + element
            if idx2 > len(data):
                element = element + ([empty_value] * (idx2 - len(data)))
            result.append(element)
        return result
