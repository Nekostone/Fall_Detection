import os
import json
import subprocess
import numpy as np
import unittest


current_dir = os.path.dirname(os.path.realpath(__file__))

class PositiveTest_1_1(unittest.TestCase):
    temp_folder_dir = os.path.join(current_dir, "temp")
    samplefalldata_dir = os.path.join(temp_folder_dir, "fall")
    samplenotfalldata_dir = os.path.join(temp_folder_dir, "not_fall")
    output_dir = os.path.join(current_dir, "../temp")

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
        subprocess.run("python3 {0} --output_filename {1} --mode {2} --sample_number {3}".format(createdata_file_dir, output_filename, 3, 1), shell=True)

        # generate sample nonfall data
        output_filename = os.path.join(self.samplenotfalldata_dir, "sample2.npy")
        subprocess.run("python3 {0} --output_filename {1} --mode {2} --sample_number {3}".format(createdata_file_dir, output_filename, 3, 1), shell=True)
    
    def test_main(self):
        program_abs_dir = os.path.join(current_dir, "../retrieve_coordinates.py")
        input_fall_dir = self.samplefalldata_dir
        input_not_fall_dir = self.samplenotfalldata_dir
        number_of_parallel_processes = 1

        subprocess.run("python3 {0} --input_fall_dir {1} --input_not_fall_dir {2} --number_of_parallel_processes {3} --is_test {4}".format(program_abs_dir, input_fall_dir, input_not_fall_dir, number_of_parallel_processes, True), shell=True)

        output_file_dir = os.path.join(self.output_dir, "0_coordinates_dict.json")
        with open(output_file_dir, "r") as readfile:
            result = json.load(readfile)

        to_compare = {
            'fall': {
                'sample1.npy': [
                    {
                        'values_list': [1.0, 1.0, 1.0], 
                        'coordinates_dict': {
                            '1.0': [[0, 0], [0, 1], [0, 2]]
                        }
                    }, 
                    {
                        'values_list': [1.0], 
                        'coordinates_dict': {
                            '1.0': [[1, 1]]}}]
            }, 
            'not_fall': {
                'sample2.npy': [
                    {
                        'values_list': [1.0, 1.0, 1.0], 
                        'coordinates_dict': {
                            '1.0': [[0, 0], [0, 1], [0, 2]]
                        }
                    }, 
                    {
                        'values_list': [1.0], 
                        'coordinates_dict': {'1.0': [[1, 1]]}
                    }
                ]
            }
        }

        self.assertEqual(result, to_compare)
    
    def tearDown(self):
        # delete sample data temp folder and contents
        if os.path.exists(self.temp_folder_dir):
            subprocess.run("rm -rf {0}".format(self.temp_folder_dir), shell=True)
        else:
            print("ERROR - sample data temp folder does not exist.")
            exit(0)

        # delete output temp folder and contents
        if os.path.exists(self.output_dir):
            subprocess.run("rm -rf {0}".format(self.output_dir), shell=True)
        else:
            print("ERROR - output temp folder does not exist.")
            exit(0)


if __name__ == "__main__":
    unittest.main()