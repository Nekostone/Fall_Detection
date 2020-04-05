import os
import subprocess
import numpy as np
import unittest


current_dir = os.path.dirname(os.path.realpath(__file__))

class PositiveTest(unittest.TestCase):
    temp_folder_dir = os.path.join(current_dir, "temp")
    sampledata_dir = os.path.join(temp_folder_dir, "sample1.npy")

    def setUp(self):
        # create temp folder
        if os.path.exists(self.temp_folder_dir):
            print("ERROR - Temp folder already exist. Please delete before running script")
            exit(0)
        os.mkdir(self.temp_folder_dir)

        # generate sample data
        createdata_file_dir = os.path.join(current_dir, "../../create_sample_data.py")
        subprocess.run("python3 {0} --output_filename {1} --mode {2} --sample_number {3}".format(createdata_file_dir, self.sampledata_dir, 2, 2), shell=True)
    
    def test_main(self):
        thresholding_dir = os.path.join(current_dir, "../thresholding.py")
        outputdata_dir = os.path.join(self.temp_folder_dir, "output1.npy")

        subprocess.run("python3 {0} --input_filename {1} --output_filename {2} --width_of_ignored_pixels {3} --width_of_rim_for_comparison {4}".format(thresholding_dir, self.sampledata_dir, outputdata_dir, 1, 1), shell=True)

        result = np.load(outputdata_dir)
        to_compare = np.array([
            [
                [0,2,3,4,3,2,0],
                [2,3,4,5,4,3,2],
                [3,4,5,6,5,4,3],
                [4,5,6,7,6,5,4],
                [3,4,5,6,5,4,3],
                [2,3,4,5,4,3,2],
                [0,2,3,4,3,2,0],
            ],
            [
                [0,2,3,4,3,2,0],
                [2,3,4,5,4,3,2],
                [3,4,5,6,5,4,3],
                [4,5,6,7,6,5,4],
                [3,4,5,6,5,4,3],
                [2,3,4,5,4,3,2],
                [0,2,3,4,3,2,0],
            ]
        ], dtype=np.float64)

        comparison_result = (result == to_compare).all()
        self.assertTrue(comparison_result)
    
    def tearDown(self):
        # delete temp folder and contents
        if os.path.exists(self.temp_folder_dir):
            subprocess.run("rm -rf {0}".format(self.temp_folder_dir), shell=True)
        else:
            print("ERROR - Temp folder does not exist.")
            exit(0)


if __name__ == "__main__":
    unittest.main()