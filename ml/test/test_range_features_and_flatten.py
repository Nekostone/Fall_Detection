import numpy as np
import unittest

from ml.feature_extract.range_features_and_flatten import range_profile, dopp_profile, range_delta, dopp_delta, range_features_and_flatten


class Range_profile(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.array([
            [
                [1,1,1],
                [2,2,2],
                [3,3,3]
            ],
            [
                [2,2,2],
                [3,3,3],
                [4,4,4]
            ],
            [
                [3,3,3],
                [4,4,4],
                [5,5,5]
            ]
        ])

        output = range_profile(input_array)
        to_compare = np.array([
            [3,6,9],
            [6,9,12],
            [9,12,15]
        ])

        self.assertTrue((output==to_compare).all())

class Dopp_profile(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.array([
            [
                [1,1,1],
                [2,2,2],
                [3,3,3]
            ],
            [
                [2,2,2],
                [3,3,3],
                [4,4,4]
            ],
            [
                [3,3,3],
                [4,4,4],
                [5,5,5]
            ]
        ])

        output = dopp_profile(input_array)
        to_compare = np.array([
            [6,6,6],
            [9,9,9],
            [12,12,12]
        ])

        self.assertTrue((output==to_compare).all())

class Range_delta(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.array([
            [3,6,9],
            [6,9,12],
            [9,12,15]
        ])

        output = range_delta(input_array)
        to_compare = np.array([[18,27,36]])

        self.assertTrue((output==to_compare).all())

class Dopp_delta(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.array([
            [6,6,6],
            [9,9,9],
            [12,12,12]
        ])

        output = dopp_delta(input_array)
        to_compare = np.array([[18,27,36]])

        self.assertTrue((output==to_compare).all())

class Range_features_and_flatten(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.array([
            np.array([
                [
                    [1,1,1],
                    [2,2,2],
                    [3,3,3]
                ],
                [
                    [2,2,2],
                    [3,3,3],
                    [4,4,4]
                ],
                [
                    [3,3,3],
                    [4,4,4],
                    [5,5,5]
                ]
            ]),
            1
        ])

        output = range_features_and_flatten(input_array)
        to_compare = np.array([
            np.array([3, 6, 9, 6, 9, 12, 9, 12, 15, 18, 27, 36, 6, 6, 6, 9, 9, 9, 12, 12, 12, 18, 27, 36]),
            1
        ])

        self.assertTrue((output[0]==to_compare[0]).all())
        self.assertEqual(output[1], to_compare[1])



if __name__ == "__main__":
    unittest.main()
