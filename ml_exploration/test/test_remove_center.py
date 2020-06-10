import unittest
import numpy as np
from ml_exploration.filters.remove_center import remove_center

class Remove_center(unittest.TestCase):
    def test_positive_0(self):
        """
        input_array = np.full((50,128,128), 1)
        input_array = np.array([input_array, 1])
        result = remove_center(input_array)

        to_compare = np.full((50,63,128), 1)
        to_concat = np.full((50,3,128),0)
        to_compare = np.concatenate((to_compare, to_concat), axis=1)
        to_concat = np.full((50,62,128), 1)
        to_compare = np.concatenate((to_compare, to_concat), axis=1)

        self.assertTrue((result == to_compare))
        """
        pass


if __name__ == '__main__':
    unittest.main()
