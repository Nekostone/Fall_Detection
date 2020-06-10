import numpy as np
import unittest

from ml_final._range_features_and_flatten_localnorm import range_profile, dopp_profile, range_features_and_flatten_localnorm


class Range_profile(unittest.TestCase):
    def test_positive_0(self):
        # range_axis = 0
        input_array = np.array([
            [
                [1,1,1],
                [2,3,2],
                [3,3,3]
            ],
            [
                [1,1,1],
                [2,3,2],
                [3,3,3]
            ]
        ])

        output = range_profile(input_array)
        to_compare = np.array([
            [6,7,6],
            [6,7,6]
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
                [1,1,1],
                [2,2,2],
                [3,3,3]
            ]
        ])

        output = dopp_profile(input_array)
        to_compare = np.array([
            [3,6,9],
            [3,6,9]
        ])

        self.assertTrue((output==to_compare).all())

class Range_features_and_flatten_localnorm(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.array([
            np.array([
                [
                    [1,1,1],
                    [2,3,2],
                    [3,3,3]
                ],
                [
                    [1,1,1],
                    [2,3,2],
                    [3,3,3]
                ]
            ]),
            1
        ])

        to_compare = np.array([
            np.array([-0.70710678, 1.41421356, -0.70710678, -0.70710678, 1.41421356, -0.70710678, -1.33630621, 0.26726124, 1.06904497, -1.33630621, 0.26726124, 1.06904497]),
            1
        ])

        output = range_features_and_flatten_localnorm(input_array)
        self.assertTrue((np.round(output[0], 3)==np.round(to_compare[0],3)).all())
        self.assertEqual(output[1], to_compare[1])



if __name__ == "__main__":
    unittest.main()
