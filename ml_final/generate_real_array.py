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
    pre_train_dir = "/home/xubuntu/Desktop/Fall_Detection/ml_final/temp"

    for each_file in os.listdir(vanilla_labelled_dir):
        each_file_dir = os.path.join(vanilla_labelled_dir, each_file)
        input_array = np.load(each_file_dir, allow_pickle=True)

        # transpose each frame
        data = input_array[0]
        print("data.shape: {0}".format(data.shape))
        for i in range(data.shape[0]):
            data[i,:,:] = np.transpose(data[i,:,:])

        # downsample_time 
        downsample_time_factor = 25
        input_array = downsample_time(input_array, downsample_time_factor)[0]  # we only need one array

        # downsampled doppler
        input_array = downsample_doppler(input_array, 2)

        # save array
        output_dir = os.path.join(pre_train_dir, "{0}.npy".format(each_file[:-4]))
        np.save(output_dir, input_array[0], allow_pickle=True)  # save only the array, not the label
        exit(0)
