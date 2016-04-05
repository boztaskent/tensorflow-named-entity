
import unittest

from detect.util.data_helpers import DataHelper


class TestDataHelpers(unittest.TestCase):

    def test(self):
        training_data = [
            ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "data/profiles/train/davidcameron_1.txt"),
            ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "data/profiles/train/davidcameron_2.txt"),
            ([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "data/profiles/train/davidcameron_3.txt"),
            ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "data/profiles/train/davidcameron_4.txt"),
            ([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "data/profiles/train/davidcameron_5.txt"),
            ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "data/profiles/train/davidcameron_6.txt")]

        # x_text, y = self.dataHelper.load_data_and_labels(training_data)

        data_helper = DataHelper()

        x, y, voc, vocInv = data_helper.load_data(training_data)

        batches = data_helper.batch_iter(x, 50, 1, shuffle=False)
        for x_test_batch in batches:
            print(len(x_test_batch))
            print(x_test_batch[0])
