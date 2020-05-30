import argparse
import os
import subprocess
import json
import numpy as np

from feature_extract.extractcoords_and_flatten import extractcoords_and_flatten
from feature_extract.energy_features_and_flatten import energy_features_and_flatten
from feature_extract.range_features_and_flatten import range_features_and_flatten 

from filters.remove_center import remove_center
from filters.downsample import downsample

from models.randomforest import randomforest
import pickle

output_dir = "/home/chongyicheng/Capstone/Fall_Detection/ml/onearrayfeatures"
input_array = np.load("/home/chongyicheng/Capstone/labelled_vanilla/20200320_jd_4m_radial3.npy",allow_pickle=True)
rfc = pickle.load(open("/home/chongyicheng/Capstone/Fall_Detection/randomforest.sav", 'rb'))
count = 0
# print(input_array[0].shape)
#downsample
downsample_factor = 2
output = downsample(input_array, downsample_factor)
#remove center
isFirst = True
features = []
for each_downsampled_output in output:
    output_without_center = remove_center(each_downsampled_output)
    #range features
    output_features = range_features_and_flatten(output_without_center)
    #print(output_features[0])
    features.append(output_features[0])
    #print("Prediction:",rfc.predict(output_features[0]))
    #print("Actual:", input_array[1])


print("Prediction:",rfc.predict(features))
print("Actual:", input_array[1])


    # output_direct = os.path.join(output_dir, "{0}.npy".format(count))
    # if isFirst == True:
    #     print(output_features.shape)
    #     isFirst = False

    # np.save(output_direct, output_features, allow_pickle=True)
    # count += 1

# for file in os.listdir(output_dir):
#     each_file = os.path.join(output_dir,file)
#     input_array = np.load(each_file,allow_pickle=True)
#     print("Prediction: ",rfc.predict(input_array[0]))
#     print("Actual: ",input_array[1])
