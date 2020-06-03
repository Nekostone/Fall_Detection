import argparse
import os
import subprocess
import json
import numpy as np

from feature_extract.extractcoords_and_flatten import extractcoords_and_flatten
from feature_extract.energy_features_and_flatten import energy_features_and_flatten
from feature_extract.range_features_and_flatten import range_features_and_flatten 
from feature_extract.range_features_and_flatten_localnorm import range_features_and_flatten_localnorm
from feature_extract.downsample_doppler import downsample_doppler

from filters.downsample_time import downsample_time
from filters.remove_center import remove_center

from models.svm import svm

from misc.normalize_to_train import normalize_to_train
from misc.range_features_normalized import range_features_normalized

from split_train_test import split_train_test


if __name__ == "__main__":
    vanilla_labelled_dir = "/home/xubuntu/Desktop/sensor_data/labelled_vanilla"
    pre_train_dir = "/home/xubuntu/Desktop/Fall_Detection/ml_exploration/temp"

    # recreate temp folder
    if os.path.exists(pre_train_dir):
        subprocess.call("rm -rf {0}".format(pre_train_dir), shell=True)
        print("ERROR - folder {0} already exist. Deleting old folder...".format(pre_train_dir))
    os.mkdir(pre_train_dir)

    count = 0
    for each_file in os.listdir(vanilla_labelled_dir):
        each_file_dir = os.path.join(vanilla_labelled_dir, each_file)
        input_array = np.load(each_file_dir, allow_pickle=True)

        # downsample_time 
        downsample_time_factor = 25
        output = downsample_time(input_array, downsample_time_factor)

        for each_downsampled_output in output:
            # remove center
            each_downsampled_output = remove_center(each_downsampled_output)

            # downsampled doppler
            each_downsampled_output = downsample_doppler(each_downsampled_output, 2)

            # extract range features and normalize across all elements in one recording
            each_downsampled_output = range_features_and_flatten_localnorm(each_downsampled_output)

            # save array
            each_downsampled_output_dir = os.path.join(pre_train_dir, "{0}.npy".format(count))
            np.save(each_downsampled_output_dir, each_downsampled_output, allow_pickle=True)
            print("{0} processed.".format(count))

            count += 1

    # SVM
    print("Starting model training and testing...")
    train_percentage = 0.6
    train_x, train_y, test_x, test_y = split_train_test(pre_train_dir, train_percentage)
    true_positive, true_negative, false_positive, false_negative = svm(train_x, train_y, test_x, test_y)
    print("true_positive: {0}, true_negative: {1}, false_positive: {2}, false_negative: {3}".format(true_positive, true_negative, false_positive, false_negative))

    # cleanup
    subprocess.call("rm -rf {0}".format(pre_train_dir), shell=True)
