import argparse
import os
import subprocess
import json
import numpy as np

from _range_features_and_flatten_localnorm import range_features_and_flatten_localnorm
from _downsample_doppler import downsample_doppler

from _downsample_time import downsample_time
from _remove_center import remove_center

from _svm import svm
from _split_train_test import split_train_test


if __name__ == "__main__":
    vanilla_labelled_dir = "/home/xubuntu/Desktop/sensor_data/labelled_vanilla"
    tm_nonfall_dir = "/home/xubuntu/Desktop/sensor_data/tm_time10_doppler2"  # y-axis doppler, x-axis range, downsampled time by 10, downsampled doppler by 2
    pre_train_dir = "/home/xubuntu/Desktop/Fall_Detection/ml_final/temp"
    weights_dir = "/home/xubuntu/Desktop/Fall_Detection/ml_final/weights.pickle"

    # recreate temp folder
    if os.path.exists(pre_train_dir):
        subprocess.call("rm -rf {0}".format(pre_train_dir), shell=True)
        print("ERROR - folder {0} already exist. Deleting old folder...".format(pre_train_dir))
    os.mkdir(pre_train_dir)

    # iterate for original data
    count = 0
    # iterate for special snowflake data
    for each_file in os.listdir(tm_nonfall_dir):
        each_file_dir = os.path.join(tm_nonfall_dir, each_file)
        input_array = np.load(each_file_dir, allow_pickle=True)

        # transpose each frame
        input_array[0]= np.moveaxis(input_array[0], 1, -1)

        input_array = remove_center(input_array, 31, 34)
        input_array = range_features_and_flatten_localnorm(input_array)

        # save array
        output_dir = os.path.join(pre_train_dir, "{0}.npy".format(count))
        np.save(output_dir, input_array, allow_pickle=True)
        print("{0} processed".format(count))

        count += 1

    for each_file in os.listdir(vanilla_labelled_dir):
        each_file_dir = os.path.join(vanilla_labelled_dir, each_file)
        input_array = np.load(each_file_dir, allow_pickle=True)

        # transpose each frame
        input_array[0]= np.moveaxis(input_array[0], 1, -1)

        # downsample_time 
        downsample_time_factor = 10
        output = downsample_time(input_array, downsample_time_factor)

        for each_downsampled_output in output:
            each_downsampled_output = remove_center(each_downsampled_output)
            each_downsampled_output = downsample_doppler(each_downsampled_output, 2)
            each_downsampled_output = range_features_and_flatten_localnorm(each_downsampled_output)

            # save array
            each_downsampled_output_dir = os.path.join(pre_train_dir, "{0}.npy".format(count))
            np.save(each_downsampled_output_dir, each_downsampled_output, allow_pickle=True)
            print("{0} processed".format(count))

            count += 1


    # SVM
    print("Starting model training and testing...")
    train_percentage = 0.9
    fall_percentage = 0.5
    train_x, train_y, test_x, test_y = split_train_test(pre_train_dir, train_percentage, fall_percentage)
    true_positive, true_negative, false_positive, false_negative = svm(train_x, train_y, test_x, test_y, weights_dir)
    print("true_positive: {0}, true_negative: {1}, false_positive: {2}, false_negative: {3}".format(true_positive, true_negative, false_positive, false_negative))

    # cleanup
    subprocess.call("rm -rf {0}".format(pre_train_dir), shell=True)
