import numpy as np

def downsample_time(input_array, downsample_factor):
    data = input_array[0]
    label = input_array[1]
    output_array = []

    if downsample_factor > data.shape[0]:
        exit("ERROR - Down sample factor is larger than the total number of frames in one recording")

    for i in range(downsample_factor):
        first_row = data[i,:,:]
        first_row_expanded = np.expand_dims(first_row, axis=0)
        output_array.append(first_row_expanded)

    state = 0
    for i in range(downsample_factor, data.shape[0]):
        next_row = data[i,:,:]
        next_row_expanded = np.expand_dims(next_row, axis=0)
        output_array[state] = np.concatenate((output_array[state], next_row_expanded), axis=0)
        
        state += 1
        if state >= downsample_factor:
            state = 0
    
    for i in range(len(output_array)):
        output_array[i] = np.array([output_array[i], label])

    return output_array
        
