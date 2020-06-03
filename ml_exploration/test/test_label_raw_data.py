"""
Run this test from root folder using `python3 -m unittest`
"""

import unittest
import numpy as np
import sys
import os
import subprocess
import csv

from ml_exploration.label_raw_data import process_fall_data, process_nonfall_data


file_dir = os.path.dirname(os.path.realpath(__file__))

class Process_fall_data(unittest.TestCase):
    def test_positive_1(self):
        nonfall_frames = np.full((1,3,3), -1)
        fall_frames = np.full((50,3,3), 0)
        fall_frames = np.concatenate((nonfall_frames, fall_frames), axis=0)
        nonfall_frames = np.full((99,3,3), -1)
        fall_frames = np.concatenate((fall_frames, nonfall_frames), axis=0)

        starting_frame = 1
        ending_frame = 51

        result = process_fall_data(fall_frames, starting_frame, ending_frame)

        to_compare = np.full((50,3,3), 0)
        to_compare = np.array([to_compare, 1])

        # we'll compare the y label and the x features of each datapoint separately (since the x features is a individual numpy array)
        comparison_result = np.equal(result[0], to_compare[0])
        if not comparison_result.all:
            print('comparison_result: {0}'.format(comparison_result))

        self.assertTrue(comparison_result.all)
        self.assertEqual(result[1], to_compare[1])

    def test_positive_2(self):
        """
        Simpler form of test_positive_1, used for debugging mostly
        """
        nonfall_frames = np.full((1,3,3), -1)
        fall_frames = np.full((2,3,3), 0)
        fall_frames = np.concatenate((nonfall_frames, fall_frames), axis=0)
        nonfall_frames = np.full((2,3,3), -1)
        fall_frames = np.concatenate((fall_frames, nonfall_frames), axis=0)

        starting_frame = 1
        ending_frame = 3

        result = process_fall_data(fall_frames, starting_frame, ending_frame)

        to_compare = np.full((2,3,3), 0)
        to_compare = np.array([to_compare, 1])

        comparison_result = np.equal(result[0], to_compare[0])
        if not comparison_result.all:
            print('comparison_result: {0}'.format(comparison_result))

        self.assertTrue(comparison_result.all)
        self.assertEqual(result[1], to_compare[1])


class Process_nonfall_data(unittest.TestCase):
    def test_positive_1(self):
        nonfall_frames = np.full((50,3,3), 0)
        to_concat = np.full((50,3,3), 1)
        nonfall_frames = np.concatenate((nonfall_frames, to_concat), axis=0)
        to_concat = np.full((50,3,3), 2)
        nonfall_frames = np.concatenate((nonfall_frames, to_concat), axis=0)

        result = process_nonfall_data(nonfall_frames)

        to_compare_0 = np.array([np.full((50,3,3),0), 0])
        to_compare_1 = np.array([np.full((50,3,3),1), 0])
        to_compare_2 = np.array([np.full((50,3,3),2), 0])

        comparison_result_0 = np.equal(result[0][0], to_compare_0[0])
        comparison_result_1 = np.equal(result[1][0], to_compare_1[0])
        comparison_result_2 = np.equal(result[2][0], to_compare_2[0])

        # compare x features
        self.assertTrue(comparison_result_0.all)
        self.assertTrue(comparison_result_1.all)
        self.assertTrue(comparison_result_2.all)

        # to compare y labels
        self.assertEqual(to_compare_0[1], result[0][1])
        self.assertEqual(to_compare_1[1], result[1][1])
        self.assertEqual(to_compare_2[1], result[2][1])

class Integration(unittest.TestCase):
    def setUp(self):
        # create temp folder to store function metadata
        self.temp_dir = os.path.join(file_dir, 'temp/')
        if os.path.exists(self.temp_dir):
            print('ERROR - Temp metadata folder already exist. Please delete before running script')
            exit(0)
        os.mkdir(self.temp_dir)

        self.temp_fall_folder_dir = os.path.join(self.temp_dir, 'fall/')
        if os.path.exists(self.temp_fall_folder_dir):
            print('ERROR - Temp metadata folder already exist. Please delete before running script')
            exit(0)
        os.mkdir(self.temp_fall_folder_dir)

        self.temp_nonfall_folder_dir = os.path.join(self.temp_dir, 'nonfall/')
        if os.path.exists(self.temp_nonfall_folder_dir):
            print('ERROR - Temp metadata folder already exist. Please delete before running script')
            exit(0)
        os.mkdir(self.temp_nonfall_folder_dir)

        # integration test will focus on the writing and reading of files, as to the logic that is used to process the data, we'll leave it to the unit tests
        fall_npy_dir = os.path.join(self.temp_fall_folder_dir, "fall_0.npy")
        self.temp_fall_output_dir = os.path.join(self.temp_dir, "fall_0.npy")
        fall_frames = np.full((150,3,3), 0)
        np.save(fall_npy_dir, fall_frames)

        nonfall_npy_dir = os.path.join(self.temp_nonfall_folder_dir, "nonfall_0.npy")
        self.temp_nonfall_output0_dir = os.path.join(self.temp_dir, "nonfall_0_0.npy")
        self.temp_nonfall_output1_dir = os.path.join(self.temp_dir, "nonfall_0_1.npy")
        self.temp_nonfall_output2_dir = os.path.join(self.temp_dir, "nonfall_0_2.npy")
        nonfall_frames = np.full((150,3,3), 0)
        np.save(nonfall_npy_dir, nonfall_frames)

        self.input_csv = os.path.join(self.temp_dir, 'labels.csv')
        with open(self.input_csv, "w+") as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"')
            spamwriter.writerow(["fall_0.npy", 1, 51])


    def test_main(self):
        source_dir = os.path.join(file_dir, '../label_raw_data.py')

        subprocess.run('python3 {0} --input_folder {1} --input_csv {2} --output_folder {3}'.format(source_dir, self.temp_fall_folder_dir, self.input_csv, self.temp_dir), shell=True)
        subprocess.run('python3 {0} --input_folder {1} --output_folder {2} --is_nonfall_data'.format(source_dir, self.temp_nonfall_folder_dir, self.temp_dir), shell=True)

        for each_dir in [self.temp_fall_output_dir, self.temp_nonfall_output0_dir, self.temp_nonfall_output1_dir, self.temp_nonfall_output2_dir]:
            input_data = np.load(each_dir, allow_pickle=True)
            self.assertIsInstance(input_data, np.ndarray)

    def tearDown(self):
        # delete sample data temp folder and contents
        if os.path.exists(self.temp_dir):
            subprocess.run('rm -rf {0}'.format(self.temp_dir), shell=True)
        else:
            print('ERROR - {0} does not exist.'.format(self.temp_dir))
            exit(0)



if __name__ == '__main__':
    unittest.main()

























