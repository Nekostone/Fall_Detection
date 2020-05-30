import unittest
import numpy as np

from ml.feature_extract.downsample_doppler import downsample_doppler

class Downsample(unittest.TestCase):
    def test_positive_0(self):
        downsample_factor = 2
        input_array = np.array([
            np.array([
                [
                    [1,1,1],
                    [2,2,2],
                    [3,3,3],
                    [4,4,4],
                    [5,5,5],
                    [6,6,6],
                    [7,7,7],
                    [8,8,8]
                ],
                [
                    [2,2,2],
                    [3,3,3],
                    [4,4,4],
                    [5,5,5],
                    [6,6,6],
                    [7,7,7],
                    [8,8,8],
                    [9,9,9]
                ],
            ]),
            1
        ])

        output_array = downsample_doppler(input_array, downsample_factor, 3)
        to_compare = np.array([
            np.array([
                [
                    [2,2,2],
                    [4,4,4],
                    [6,6,6],
                    [8,8,8]
                ],
                [
                    [3,3,3],
                    [5,5,5],
                    [7,7,7],
                    [9,9,9]
                ]
            ]),
            1
        ])

        self.assertTrue((output_array[0] == to_compare[0]).all)
        self.assertEqual(output_array[1], to_compare[1])

    def test_positive_1(self):
        downsample_factor = 2
        input_array = np.array([
            np.array([
                [
                    [1,1,1],
                    [2,2,2],
                    [3,3,3],
                    [4,4,4],
                    [5,5,5],
                    [6,6,6],
                    [7,7,7],
                    [8,8,8],
                    [9,9,9]
                ],
                [
                    [2,2,2],
                    [3,3,3],
                    [4,4,4],
                    [5,5,5],
                    [6,6,6],
                    [7,7,7],
                    [8,8,8],
                    [9,9,9],
                    [10,10,10]
                ],
            ]),
            1
        ])

        output_array = downsample_doppler(input_array, downsample_factor, 4)
        to_compare = np.array([
            np.array([
                [
                    [1,1,1],
                    [3,3,3],
                    [5,5,5],
                    [7,7,7],
                    [9,9,9]
                ],
                [
                    [2,2,2],
                    [4,4,4],
                    [6,6,6],
                    [8,8,8],
                    [10,10,10]
                ]
            ]),
            1
        ])

        self.assertTrue((output_array[0] == to_compare[0]).all)
        self.assertEqual(output_array[1], to_compare[1])

    def test_positive_2(self):
        downsample_factor = 3
        input_array = np.array([
            np.array([
                [
                    [1,1,1],
                    [2,2,2],
                    [3,3,3],
                    [4,4,4],
                    [5,5,5],
                    [6,6,6],
                    [7,7,7],
                    [8,8,8]
                ],
                [
                    [2,2,2],
                    [3,3,3],
                    [4,4,4],
                    [5,5,5],
                    [6,6,6],
                    [7,7,7],
                    [8,8,8],
                    [9,9,9]
                ],
            ]),
            1
        ])

        output_array = downsample_doppler(input_array, downsample_factor, 4)
        to_compare = np.array([
            np.array([
                [
                    [2,2,2],
                    [5,5,5],
                    [8,8,8]
                ],
                [
                    [3,3,3],
                    [6,6,6],
                    [9,9,9]
                ]
            ]),
            1
        ])

        self.assertTrue((output_array[0] == to_compare[0]).all)
        self.assertEqual(output_array[1], to_compare[1])

if __name__ == "__main__":
    unittest.main()
