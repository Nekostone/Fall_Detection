import unittest
import numpy as np

from ml_final._downsample_doppler import downsample_doppler

class Downsample(unittest.TestCase):
    def test_positive_0(self):
        downsample_factor = 2
        input_array = np.array([
            np.array([
                [
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5]
                ],
                [
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5]
                ],
            ]),
            1
        ])

        output_array = downsample_doppler(input_array, downsample_factor, 2)
        to_compare = np.array([
            np.array([
                [
                    [1,3,5],
                    [1,3,5],
                    [1,3,5]
                ],
                [
                    [1,3,5],
                    [1,3,5],
                    [1,3,5]
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
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5]
                ],
                [
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5]
                ],
            ]),
            1
        ])

        output_array = downsample_doppler(input_array, downsample_factor, 3)
        to_compare = np.array([
            np.array([
                [
                    [2,4],
                    [2,4],
                    [2,4]
                ],
                [
                    [2,4],
                    [2,4],
                    [2,4]
                ]
            ]),
            1
        ])

        self.assertTrue((output_array[0] == to_compare[0]).all)
        self.assertEqual(output_array[1], to_compare[1])


if __name__ == "__main__":
    unittest.main()
