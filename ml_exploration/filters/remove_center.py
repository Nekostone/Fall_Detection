import numpy as np


def remove_center(input_array, start_column=63, end_column=66):
    data = input_array[0]
    altered_data = data[:,:start_column,:]
    number_of_frames = data.shape[0]
    to_concat = np.full((number_of_frames,end_column-start_column,data.shape[2]), 0)
    altered_data = np.concatenate((altered_data, to_concat), axis=1)
    altered_data = np.concatenate((altered_data, data[:,end_column:,:]), axis=1)

    return np.array([altered_data, input_array[1]])
