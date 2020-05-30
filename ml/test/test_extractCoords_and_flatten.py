import numpy as np
import unittest

from ml.feature_extract.extractcoords_and_flatten import extractcoords_and_flatten


class ExtractCoords_and_flatten(unittest.TestCase):
    def test_positive_0(self):
        input_array = np.full((2,3,3), 1)
        input_array[0,0,0] = 2
        input_array[0,1,0] = 2
        input_array[1,0,1] = 2
        input_array[1,2,0] = 2
        input_array = np.array([input_array, 1])

        extracted_per_frame = 2

        output_array = extractcoords_and_flatten(input_array, extracted_per_frame)
        to_compare_0 = np.array([1, 0, 0, 0, 2, 0, 0, 1])  # in each frame, coordinates of the further elements are added to output first

        self.assertTrue((output_array[0]==to_compare_0).all())
        self.assertTrue(output_array[1]==1)

if __name__ == "__main__":
    unittest.main()
