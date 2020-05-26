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

    each_file = "20200319_sw_3m_sideways_occ.npy"
    each_file_dir = os.path.join(vanilla_labelled_dir, each_file)
    input_array = np.load(each_file_dir, allow_pickle=True)

    range_features_and_flatten(input_array)
