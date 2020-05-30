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
actual_input_array = np.load("/home/chongyicheng/Capstone/labelled_vanilla/20200320_jd_4m_radial3.npy",allow_pickle=True)

#Load the randomforest model
rfc = pickle.load(open("/home/chongyicheng/Capstone/Fall_Detection/randomforest_iter2.sav", 'rb'))


#This portion is just to shape the incoming data so that it can be parsed through the methods below
input_array_without_label = actual_input_array[0]
input_array = np.array([input_array_without_label, 0])

#downsample
downsample_time_factor = 25
output = downsample_time(input_array, downsample_time_factor)


data_to_be_predicted = ""
for each_downsampled_output in output:
    #remove center
    each_downsampled_output = remove_center(each_downsampled_output)

    # downsampled doppler
    each_downsampled_output = downsample_doppler(each_downsampled_output, 2)

    #range features
    final_output = range_features_and_flatten(each_downsampled_output)
    #print("Output feature on randomforestlivetest")
    #print(len(final_output[0]))
    data_to_be_predicted = final_output
    #print("Prediction:",rfc.predict(output_features[0]))
    #print("Actual:", input_array[1])


prediction = rfc.predict([data_to_be_predicted[0]])

if prediction == 1:
    print("Prediction: FALL")
elif prediction == 0:
    print("Prediction: NOT FALL")
else:
    print("Prediction: Unable to predict")
    print("Prediction value is",prediction)




