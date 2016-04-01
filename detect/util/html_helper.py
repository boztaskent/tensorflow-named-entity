# Copyright (C) 2016 Thomson Reuters. All Rights Reserved.
#

import re
from HTMLParser import HTMLParser

import requests


class HTMLStripper(HTMLParser):
    """
    Strip HTML tags and return only the raw data within the HTML content.
    """

    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


class HtmlHelper(object):
    """
    A set of helper methods for reading and parsing HTML.
    """

    @staticmethod
    def read_url(url):
        """
        Read the contents of a URL and return as a string
        """
        return requests.get(url).content

    @staticmethod
    def strip_html(data):
        """
        Strip the HTML tags from a web page of data and return the content.
        """
        data = data.replace("\n", " ")
        data = re.sub("<script.*?script>", "", data, 0)
        data = re.sub("<style.*?style>", "", data, 0)
        html_parser = HTMLStripper()
        html_parser.feed(data)
        return html_parser.get_data()
