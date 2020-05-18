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
            {
                1: [
                    [1,2],
                    [1,3]
                ],
                2: [
                    [2,1],
                    [2,2]
                ]
            },
            {
                3: [
                    [3,1],
                    [3,2]
                ]
            }
        ])
    
    def test_main(self):
        parent_dir = os.path.join(current_dir, "../")
        sys.path.insert(0, parent_dir)
        import svm

        result = svm.flatten_array(self.sampledata)
        # print("result: {0}".format(result))
        to_compare = np.array([1, 2, 1, 3, 2, 1, 2, 2, 3, 1, 3, 2])
        compare_result = (result == to_compare).all()

        self.assertTrue(compare_result)
    
if __name__ == "__main__":
    unittest.main()