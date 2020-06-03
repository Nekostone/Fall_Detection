import unittest
import numpy as np

from ml_exploration.misc.normalize_to_train import normalize_to_train

class Downsample(unittest.TestCase):
    def test_positive_0(self):
        train_x = [
            np.array([
                [
                    [1,1,1],
                    [1,1,1],
                    [1,1,1]
                ],
                [
                    [2,2,2],
                    [2,2,2],
                    [2,2,2]
                ],
                [
                    [3,3,3],
                    [3,3,3],
                    [3,3,3]
                ]
            ]),
            np.array([
                [
                    [1,1,1],
                    [1,1,1],
                    [1,1,1]
                ],
                [
                    [2,2,2],
                    [2,2,2],
                    [2,2,2]
                ],
                [
                    [3,3,3],
                    [3,3,3],
                    [3,3,3]
                ]
            ])
        ]
        train_y = [1,1]
        test_x = [
            np.array([
                [
                    [1,1,1],
                    [1,1,1],
                    [1,1,1]
                ],
                [
                    [1,1,1],
                    [1,1,1],
                    [1,1,1]
                ],
                [
                    [1,1,1],
                    [1,1,1],
                    [1,1,1]
                ]
            ])
        ]
        test_y = [1]
        result_train_x, result_train_y, result_test_x, result_test_y = normalize_to_train(train_x, train_y, test_x, test_y)

        result_train_x = np.around(result_train_x, decimals=5)
        result_test_x = np.around(result_test_x, decimals=5)

        to_compare_train_x = [
            np.array([
                [
                    [-1.22474, -1.22474, -1.22474],
                    [-1.22474, -1.22474, -1.22474],
                    [-1.22474, -1.22474, -1.22474]
                ],
                [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]
                ],
                [
                    [1.22474, 1.22474, 1.22474],
                    [1.22474, 1.22474, 1.22474],
                    [1.22474, 1.22474, 1.22474]
                ]
            ]),
            np.array([
                [
                    [-1.22474, -1.22474, -1.22474],
                    [-1.22474, -1.22474, -1.22474],
                    [-1.22474, -1.22474, -1.22474]
                ],
                [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]
                ],
                [
                    [1.22474, 1.22474, 1.22474],
                    [1.22474, 1.22474, 1.22474],
                    [1.22474, 1.22474, 1.22474]
                ]
            ])
        ]
        self.assertTrue((result_train_x == to_compare_train_x).all())

        to_compare_test_x = [np.array([
            [
                [-1.22474, -1.22474, -1.22474],
                [-1.22474, -1.22474, -1.22474],
                [-1.22474, -1.22474, -1.22474]
            ],
            [
                [-1.22474, -1.22474, -1.22474],
                [-1.22474, -1.22474, -1.22474],
                [-1.22474, -1.22474, -1.22474]
            ],
            [
                [-1.22474, -1.22474, -1.22474],
                [-1.22474, -1.22474, -1.22474],
                [-1.22474, -1.22474, -1.22474]
            ]
        ])]
        self.assertTrue((result_test_x[0] == to_compare_test_x[0]).all())


if __name__ == "__main__":
    unittest.main()
