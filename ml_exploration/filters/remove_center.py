import numpy as np


def remove_center(input_array):
    data = input_array[0]
    altered_data = data[:,:63,:]
    number_of_frames = data.shape[0]
    to_concat = np.full((number_of_frames,3,128), 0)
    altered_data = np.concatenate((altered_data, to_concat), axis=1)
    altered_data = np.concatenate((altered_data, data[:,66:,:]), axis=1)

    return np.array([altered_data, input_array[1]])
