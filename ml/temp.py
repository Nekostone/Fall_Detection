import argparse
import os
import subprocess
import json
import numpy as np
import time

from feature_extract.extractcoords_and_flatten import extractcoords_and_flatten
from feature_extract.energy_features_and_flatten import energy_features_and_flatten
from feature_extract.range_features_and_flatten import range_features_and_flatten

from filters.downsample_time import downsample_time
from filters.remove_center import remove_center

from models.svm import svm



if __name__ == "__main__":

    x = np.array([
        [1,2,3],
        [3,1,1],
        [2,1,1]
    ])
    y = [
        [4],
        [5],
        [6]
    ]
    print(np.insert(x,[0],y,axis=1))
