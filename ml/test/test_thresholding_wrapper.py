import os
import subprocess
import numpy as np
import unittest


current_dir = os.path.dirname(os.path.realpath(__file__))

class PositiveTest_1_1(unittest.TestCase):
    temp_folder_dir = os.path.join(current_dir, "temp")
    samplefalldata_dir = os.path.join(temp_folder_dir, "fall")
    samplenotfalldata_dir = os.path.join(temp_folder_dir, "not_fall")

    def setUp(self):
        # create temp folder
        if os.path.exists(self.temp_folder_dir):
            print("ERROR - Temp folder already exist. Please delete before running script")
            exit(0)
        os.mkdir(self.temp_folder_dir)

        # create temp falldata folder
        if os.path.exists(self.samplefalldata_dir):
            print("ERROR - Temp falldata folder already exist. Please delete before running script")
            exit(0)
        os.mkdir(self.samplefalldata_dir)

        # create temp nonfalldata folder
        if os.path.exists(self.samplenotfalldata_dir):
            print("ERROR - Temp nonfalldata folder already exist. Please delete before running script")
            exit(0)
        os.mkdir(self.samplenotfalldata_dir)

        createdata_file_dir = os.path.join(current_dir, "../../create_sample_data.py")

        # generate sample fall data
        output_filename = os.path.join(self.samplefalldata_dir, "sample1.npy")
        subprocess.run("python3 {0} --output_filename {1} --mode {2} --sample_number {3}".format(createdata_file_dir, output_filename, 2, 2), shell=True)

        # generate sample nonfall data
        output_filename = os.path.join(self.samplenotfalldata_dir, "sample2.npy")
        subprocess.run("python3 {0} --output_filename {1} --mode {2} --sample_number {3}".format(createdata_file_dir, output_filename, 2, 2), shell=True)
    
    def test_main(self):
        program_dir = os.path.join(current_dir, "../thresholding_wrapper.py")
        program_abs_dir = os.path.join(current_dir, "../thresholding.py")
        input_fall_dir = self.samplefalldata_dir
        input_not_fall_dir = self.samplenotfalldata_dir
        output_dir = self.temp_folder_dir
        width_of_rim_for_comparison = 1
        width_of_ignored_pixels = 1
        number_of_parallel_processes = 1

        subprocess.run("python3 {0} --program_abs_dir {1} --input_fall_dir {2} --input_not_fall_dir {3} --output_dir {4} --width_of_rim_for_comparison {5} --width_of_ignored_pixels {6} --number_of_parallel_processes {7}".format(program_dir, program_abs_dir, input_fall_dir, input_not_fall_dir, output_dir, width_of_rim_for_comparison, width_of_ignored_pixels, number_of_parallel_processes), shell=True)

        output_falldata_dir = os.path.join(self.temp_folder_dir, "1_1/fall/sample1.npy")
        output_nonfalldata_dir = os.path.join(self.temp_folder_dir, "1_1/not_fall/sample2.npy")

        result1 = np.load(output_falldata_dir)
        result2 = np.load(output_nonfalldata_dir)
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

        comparison_result1 = (result1 == to_compare).all()
        comparison_result2 = (result2 == to_compare).all()
        self.assertEqual(comparison_result1, comparison_result2)
    
    def tearDown(self):
        # delete temp folder and contents
        if os.path.exists(self.temp_folder_dir):
            subprocess.run("rm -rf {0}".format(self.temp_folder_dir), shell=True)
        else:
            print("ERROR - Temp folder does not exist.")
            exit(0)


if __name__ == "__main__":
    unittest.main()