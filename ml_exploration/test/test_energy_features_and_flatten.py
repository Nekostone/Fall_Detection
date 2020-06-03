import numpy as np
import unittest

from ml_exploration.feature_extract.energy_features_and_flatten import frame_energy, delta_energy, energy_temporal_derivative, energy_features_and_flatten


class Frame_energy(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.array([
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
            ],
        ])

        output = frame_energy(input_array)
        to_compare = np.array([9, 18, 27])

        self.assertTrue((output==to_compare).all())

class Delta_energy(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.array([5,4,3,2,1])
        output = delta_energy(input_array)
        to_compare = np.array([4])

        self.assertTrue((output==to_compare).all())

class Energy_temporal_derivative(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.array([16,8,4,2,1])
        output = energy_temporal_derivative(input_array)
        to_compare = np.array([8,4,2,1])

        self.assertTrue((output==to_compare).all())

class Energy_features_and_flatten(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.array([
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
                ],
            ]),
            1
        ])

        output = energy_features_and_flatten(input_array)
        to_compare = np.array([
            np.array([9,18,27,-18,-9,-9]), 
            1
        ])

        self.assertTrue((output[0]==to_compare[0]).all())
        self.assertEqual(output[1], to_compare[1])

if __name__ == "__main__":
    unittest.main()
