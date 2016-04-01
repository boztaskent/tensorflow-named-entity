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

from google import search


class GoogleSearch(object):
    """
    This class provides functionality to perform a search on Google and then retrieve
    the urls of the top result.
    """

    def search(self, query, max_results=100):
        """
        Perform a Google search, this method will always perform a sleep before calling Google
        to ensure we don't get blacklisted.
        """
        results = list(search(query, stop=max_results, pause=2))
        return results[0:max_results]
