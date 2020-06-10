import unittest
import numpy as np
import subprocess
import os

from ml_final._split_train_test import split_train_test

file_dir = os.path.dirname(os.path.realpath(__file__))

class Split_train_test(unittest.TestCase):
    def setUp(self):
        # create temp folder to store function metadata
        self.temp_dir = os.path.join(file_dir, 'temp/')
        if os.path.exists(self.temp_dir):
            print('ERROR - Deleting existing temp metadata folder...')
            subprocess.run(["rm", "-rf", self.temp_dir])
        os.mkdir(self.temp_dir)

    def test_positive_0(self):
        fall_data = np.array([
            np.array([1,1,1]),
            1
        ])

        nonfall_data = np.array([
            np.array([0,0,0]),
            0
        ])

        count = 0
        for data in [fall_data, nonfall_data]:
            for i in range(10):
                data_dir = os.path.join(self.temp_dir, "{0}.npy".format(count))
                np.save(data_dir, data, allow_pickle=True)
                count += 1

        train_percentage = 0.6
        fall_percentage = 0.5
        train_x, train_y, test_x, test_y = split_train_test(self.temp_dir, train_percentage, fall_percentage)
        
        train_fall_count_x = 0
        train_fall_count_y = 0
        for each_array in train_x:
            if each_array[0] == 1:
                train_fall_count_x += 1
        for each_label in train_y:
            if each_label == 1:
                train_fall_count_y += 1

        output_fall_percentage_x = train_fall_count_x/len(train_x)
        output_fall_percentage_y = train_fall_count_y/len(train_y)
        self.assertEqual(output_fall_percentage_x, fall_percentage)
        self.assertEqual(output_fall_percentage_y, fall_percentage)

        output_train_percentage_x = len(train_x)/(len(train_x)+len(test_x))
        output_train_percentage_y = len(train_y)/(len(train_y)+len(test_y))
        self.assertEqual(output_train_percentage_x, train_percentage)
        self.assertEqual(output_train_percentage_y, train_percentage)
            

    def tearDown(self):
        # delete sample data temp folder and contents
        if os.path.exists(self.temp_dir):
            subprocess.run('rm -rf {0}'.format(self.temp_dir), shell=True)
        else:
            print('ERROR - {0} does not exist.'.format(self.temp_dir))
            exit(0)


if __name__ == "__main__":
    unittest.main()
