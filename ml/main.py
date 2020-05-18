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

from filters.downsample import downsample
from filters.remove_center import remove_center

from models.svm import svm



if __name__ == "__main__":
    vanilla_labelled_dir = "/home/xubuntu/Desktop/sensor_data/labelled_vanilla"
    pre_train_dir = "/home/xubuntu/Desktop/sensor_data/temp"
   
    # DOWNSAMPLE 
    print("Starting downsampling...")
    post_downsample = []
    for each_file in os.listdir(vanilla_labelled_dir):
        each_file_dir = os.path.join(vanilla_labelled_dir, each_file)

        input_array = np.load(each_file_dir, allow_pickle=True)
        downsample_factor = 2

        output = downsample(input_array, downsample_factor)
        post_downsample.extend(output)

    # REMOVE CENTER
    print("Starting remove_center...")
    post_remove_center = []
    for each_array in post_downsample:
        output = remove_center(each_array)
        post_remove_center.append(output)
    post_downsample.clear()

    # EXTRACT COORDS AND FLATTEN
    print("Starting extratcoords_and_flatten...")
    count = 0
    for each_array in post_remove_center:
        extracted_per_frame = 200
        output = extractcoords_and_flatten(each_array, extracted_per_frame)
        output_dir = os.path.join(pre_train_dir, "{0}.npy".format(count))
        np.save(output_dir, output, allow_pickle=True)

        count += 1
    post_remove_center.clear()
    
    # SVM
    print("Starting model training and testing...")
    train_percentage = 0.6
    true_positive, true_negative, false_positive, false_negative = svm(pre_train_dir, train_percentage)
    print("true_positive: {0}, true_negative: {1}, false_positive: {2}, false_negative: {3}".format(true_positive, true_negative, false_positive, false_negative))
