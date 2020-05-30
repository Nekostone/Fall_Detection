import unittest
import numpy as np
import os
import subprocess

from ml.split_train_test import split_train_test

file_dir = os.path.dirname(os.path.realpath(__file__))

class Split_train_test(unittest.TestCase):
    def setUp(self):
        # create temp folder to store function metadata
        self.temp_dir = os.path.join(file_dir, 'temp/')
        if os.path.exists(self.temp_dir):
            print('ERROR - Temp metadata folder already exist. Please delete before running script')
            exit(0)
        os.mkdir(self.temp_dir)

        # create input data
        fall_0_dir = os.path.join(self.temp_dir, "fall0.npy")
        self.data_to_save = np.full((5), 1)
        self.data_to_save = np.array([self.data_to_save, 1])
        np.save(fall_0_dir, self.data_to_save, allow_pickle=True)

        fall_1_dir = os.path.join(self.temp_dir, "fall1.npy")
        np.save(fall_1_dir, self.data_to_save, allow_pickle=True)

        fall_2_dir = os.path.join(self.temp_dir, "fall2.npy")
        np.save(fall_2_dir, self.data_to_save, allow_pickle=True)

        fall_3_dir = os.path.join(self.temp_dir, "fall3.npy")
        np.save(fall_3_dir, self.data_to_save, allow_pickle=True)

        nonfall_0_dir = os.path.join(self.temp_dir, "nonfall0.npy")
        self.data_to_save[1] = 0
        np.save(nonfall_0_dir, self.data_to_save, allow_pickle=True)

        nonfall_1_dir = os.path.join(self.temp_dir, "nonfall1.npy")
        np.save(nonfall_1_dir, self.data_to_save, allow_pickle=True)

        nonfall_2_dir = os.path.join(self.temp_dir, "nonfall2.npy")
        np.save(nonfall_2_dir, self.data_to_save, allow_pickle=True)

        nonfall_3_dir = os.path.join(self.temp_dir, "nonfall3.npy")
        np.save(nonfall_3_dir, self.data_to_save, allow_pickle=True)


    def test_positive_1(self):
        train_x, train_y, test_x, test_y = split_train_test(self.temp_dir, 0.75)

        # check if train and test datasets have the correct number of datapoints
        self.assertEqual(len(train_x), 6)
        self.assertEqual(len(test_x), 2)
        self.assertEqual(len(train_y), 6)
        self.assertEqual(len(test_y), 2)

        # check if train and test data have the correct features
        self.assertTrue((train_x[0]==self.data_to_save[0]).all())
        self.assertTrue((test_x[0]==self.data_to_save[0]).all())

        # check if train test labels have the correct number of each label
        self.assertEqual(train_y.count(1), 3)
        self.assertEqual(train_y.count(0), 3)
        self.assertEqual(test_y.count(1), 1)
        self.assertEqual(test_y.count(0), 1)
        

    def tearDown(self):
        # delete sample data temp folder and contents
        if os.path.exists(self.temp_dir):
            subprocess.run('rm -rf {0}'.format(self.temp_dir), shell=True)
        else:
            print('ERROR - {0} does not exist.'.format(self.temp_dir))
            exit(0)



if __name__ == "__main__":
    unittest.main()
