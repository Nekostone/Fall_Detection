import argparse
import os
import subprocess
import json
import numpy as np
import pickle


from feature_extract.range_features_and_flatten import range_features_and_flatten 
from feature_extract.downsample_doppler import downsample_doppler

from models.randomforest import randomforest

from filters.downsample_time import downsample_time
from filters.remove_center import remove_center

#Replace with the numpy array of the data you are putting in
# actual_input_array = np.load("/home/chongyicheng/Capstone/2fpstest.npy",allow_pickle=True)

input_array = np.load("")
#Load the randomforest model. Replace with the directory where the model is 
# print("Actual input array:")
# print(actual_input_array)
rfc = pickle.load(open("/home/chongyicheng/Capstone/Fall_Detection/randomforest_iter2.sav", 'rb'))


#This portion is just to shape the incoming data so that it can be parsed through the methods below

# input_array_without_label = actual_input_array[0]
# input_array = np.array([input_array_without_label, 0])

#remove center
each_downsampled_output = remove_center(input_array)
#range features
final_output = range_features_and_flatten(each_downsampled_output)

prediction = rfc.predict([final_output[0]])

if prediction == 1:
    print("Prediction: FALL")
elif prediction == 0:
    print("Prediction: NOT FALL")
else:
    print("Prediction: Unable to predict")
    print("Prediction value is",prediction)
