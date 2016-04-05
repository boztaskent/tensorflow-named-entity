
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

        x, y, voc, vocInv = data_helper.load_data(training_data, 50)
        print(vocInv)

        batches = data_helper.batch_iter(x, y, 50, 1, shuffle=True)
        for batch_x, batch_y in batches:
            print(batch_y[5, :])
            temp = [vocInv[i] for i in batch_x[5, :]]
            print(temp)