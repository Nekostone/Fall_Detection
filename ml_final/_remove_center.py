import numpy as np


def remove_center(input_array, column_to_start_removing=63, column_to_stop_removing=66):
    data = input_array[0]
    altered_data = data[:,:,:column_to_start_removing]
    to_concat = np.full((data.shape[0],data.shape[1], column_to_stop_removing-column_to_start_removing), 0)
    altered_data = np.concatenate((altered_data, to_concat), axis=2)
    altered_data = np.concatenate((altered_data, data[:,:,column_to_stop_removing:]), axis=2)

    return np.array([data, input_array[1]])
