import unittest
import numpy as np

from ml_exploration.filters.remove_center import remove_center

class Remove_center(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.array([
            np.array([
                [
                    [0,0,0],
                    [1,1,1],
                    [0,0,0]
                ],
                [
                    [0,0,0],
                    [1,1,1],
                    [0,0,0]
                ],
                [
                    [0,0,0],
                    [1,1,1],
                    [0,0,0]
                ]
            ]),
            1
        ])

        output = remove_center(input_array,1,2)
        to_compare = np.array([
            np.array([
                [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]
                ],
                [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]
                ],
                [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]
                ]
            ]),
            1
        ])

        self.assertTrue((output[0] == to_compare[0]).all())

if __name__ == "__main__":
    unittest.main()
