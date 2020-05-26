"""
Subtypes of each phase
- filters
  - downsample
  - remove_center
- feature extraction
  - extractcoords_and_flatten
- model
  - svm

config.json example:
{
    filters: [
        {
            name: downsample,
            by_factor_of: 2,
            remember: True
        },
        {
            name: remove_center,
            remember: True
        }
    ],
    feature_extraction: [
        {
            name: extractcoords_and_flatten,
            number_of_coords: 200,
            remember: False
        }
    ],
    model: [
        {
            name: svm
        }
    ]
}

"""

import argparse
import os
import subprocess
import json
import numpy as np

from feature_extract.extractcoords_and_flatten import extractcoords_and_flatten
from feature_extract.energy_features_and_flatten import energy_features_and_flatten
from feature_extract.range_features_and_flatten import range_features_and_flatten 

from filters.downsample import downsample
from filters.remove_center import remove_center

from models.svm import svm



if __name__ == "__main__":
    vanilla_labelled_dir = "/home/xubuntu/Desktop/sensor_data/labelled_vanilla"
    pre_train_dir = "/home/xubuntu/Desktop/Fall_Detection/ml/temp"

    # recreate temp folder
    if os.path.exists(pre_train_dir):
        subprocess.call("rm -rf {0}".format(pre_train_dir), shell=True)
        print("ERROR - folder {0} already exist. Deleting old folder...".format(pre_train_dir))
    os.mkdir(pre_train_dir)

    count = 0
    total_count = len(os.listdir(vanilla_labelled_dir))
    for each_file in os.listdir(vanilla_labelled_dir):
        each_file_dir = os.path.join(vanilla_labelled_dir, each_file)
        input_array = np.load(each_file_dir, allow_pickle=True)

        # DOWNSAMPLE 
        downsample_factor = 2
        output = downsample(input_array, downsample_factor)

        for each_downsampled_output in output:
            # REMOVE CENTER
            each_downsampled_output = remove_center(each_downsampled_output)

            # range features
            output = range_features_and_flatten(input_array)
            output_dir = os.path.join(pre_train_dir, "{0}.npy".format(count))
            np.save(output_dir, output, allow_pickle=True)

            print("{0}/{1} processed.".format(count, total_count))

            count += 1

    # SVM
    print("Starting model training and testing...")
    train_percentage = 0.6
    true_positive, true_negative, false_positive, false_negative = svm(pre_train_dir, train_percentage)
    print("true_positive: {0}, true_negative: {1}, false_positive: {2}, false_negative: {3}".format(true_positive, true_negative, false_positive, false_negative))
