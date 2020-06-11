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
from _feature_defs import feature_defs


def preprocess(input_array):
    frames = []
    for frame in range(len(input_array)):
        frames.append(np.expand_dims(input_array[frame], axis=0))

    input_array = np.concatenate(frames, axis=0)

    # i'm slapping some random label so that i can reuse code lol
    input_array = np.array([input_array, -1])

    # remove center
    input_array = remove_center(input_array,32,35)

    # extract range features and normalize across all elements in one recording
    input_array = feature_defs(input_array)

    return [input_array[0]]
