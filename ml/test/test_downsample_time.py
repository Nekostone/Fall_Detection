import unittest
import numpy as np

from ml.filters.downsample_time import downsample_time

class Downsample(unittest.TestCase):
    def test_positive_0(self):
        downsample_factor = 2
        is_first = True

        for repeat_pattern in range(2):
            if is_first:
                input_array = np.full((1,3,3), -1)
                is_first = False
            else:
                to_concatenate = np.full((1,3,3), -1)
                input_array = np.concatenate((input_array, to_concatenate), axis=0)

            for i in range(downsample_factor-1):
                to_concatenate = np.full((1,3,3), i)
                input_array = np.concatenate((input_array, to_concatenate), axis=0)
        input_array = np.array([input_array, 1])

        output_array = downsample_time(input_array, downsample_factor)

        to_compare_0 = np.full((2,3,3), -1)
        to_compare_1 = np.full((2,3,3), 0)

        self.assertTrue((output_array[0][0]==to_compare_0).all)
        self.assertTrue((output_array[1][0]==to_compare_0).all)
        self.assertTrue(output_array[0][1] == 1)
        self.assertTrue(output_array[1][1] == 1)

if __name__ == "__main__":
    unittest.main()
