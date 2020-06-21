import argparse
import os
import subprocess
import json
import numpy as np

from feature_extract.extractcoords_and_flatten import extractcoords_and_flatten
from feature_extract.energy_features_and_flatten import energy_features_and_flatten
from feature_extract.range_features_and_flatten import range_features_and_flatten 
from feature_extract.downsample_doppler import downsample_doppler

from filters.downsample_time import downsample_time
from filters.remove_center import remove_center

from models.svm import svm
from models.randomforest import randomforest

# from misc.normalize_to_train import normalize_to_train
# from misc.energy_features_normalized import energy_features_normalized
# from misc.range_features_normalized import range_features_normalized

from split_train_test import split_train_test


if __name__ == "__main__":
    # vanilla_labelled_dir = "/home/xubuntu/Desktop/sensor_data/labelled_vanilla"
    # pre_train_dir = "/home/xubuntu/Desktop/Fall_Detection/ml/temp"
    vanilla_labelled_dir = "/home/chongyicheng/Capstone/labelled_vanilla"
    pre_train_dir = "/home/chongyicheng/Capstone/Fall_Detection/ml/temp"


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
        # downsample_time_factor = 10
        downsample_time_factor = 25
        output = downsample_time(input_array, downsample_time_factor)

        for each_downsampled_output in output:
            
            # downsampled doppler(2)
            each_downsampled_output = downsample_doppler(each_downsampled_output, 2)
            # remove center(1)
            each_downsampled_output = remove_center(each_downsampled_output)

            #range features
            final_output = range_features_and_flatten(each_downsampled_output)
            # save array
            #each_downsampled_output_dir = os.path.join(pre_train_dir, "{0}.npy".format(count))
            #np.save(each_downsampled_output_dir, each_downsampled_output, allow_pickle=True)
            final_output_dir = os.path.join(pre_train_dir, "{0}.npy".format(count))
            np.save(final_output_dir,final_output,allow_pickle=True)
            print("{0} processed.".format(count))

            count += 1


    # # SVM
    # print("Starting model training and testing...")
    # train_percentage = 0.6
    # train_x, train_y, test_x, test_y = split_train_test(pre_train_dir, train_percentage)
    # train_x, test_x = range_features_normalized(train_x, test_x)
    # true_positive, true_negative, false_positive, false_negative = svm(train_x, train_y, test_x, test_y)
    # print("true_positive: {0}, true_negative: {1}, false_positive: {2}, false_negative: {3}".format(true_positive, true_negative, false_positive, false_negative))

    # cleanup
    #subprocess.call("rm -rf {0}".format(pre_train_dir), shell=True)
    # Random Forest
    print("Starting Random Forest Classification training and testing")
    train_percentage = 0.6
    train_x, train_y, test_x, test_y = split_train_test(pre_train_dir, train_percentage)
    true_positive, true_negative, false_positive, false_negative = randomforest(train_x, train_y, test_x, test_y)
    print("true_positive: {0}, true_negative: {1}, false_positive: {2}, false_negative: {3}".format(true_positive, true_negative, false_positive, false_negative))

