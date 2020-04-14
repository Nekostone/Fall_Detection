import os
import sys
import subprocess
import numpy as np
import unittest


current_dir = os.path.dirname(os.path.realpath(__file__))

class FlattenArrays_function_positive(unittest.TestCase):
    sampledata = None

    def setUp(self):
        self.sampledata = np.array([
            [
                'sample1.npy', 
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]]
            ],
            [
                'sample2.npy', 
                [[11, 12], [13, 14]],
                [[15, 16], [17, 18]]
            ]
        ])
    
    def test_main(self):
        parent_dir = os.path.join(current_dir, "../")
        sys.path.insert(0, parent_dir)
        import svm

        result = svm.flatten_array(self.sampledata)
        to_compare = np.array([[1,2,3,4,5,6,7,8], [11,12,13,14,15,16,17,18]])

        self.assertTrue((result==to_compare).all())
    
if __name__ == "__main__":
    unittest.main()