import numpy as np

def downsample_doppler(input_array, downsample_factor, center_of_bright_line=64):
    """
    Taking input_array where frame is 128*128, remove each alternate column radiating from the center of the center bright line of the array
    (ml_final)
    """
    data = input_array[0]
    label = input_array[1]

    if downsample_factor > data.shape[1]:
        exit("ERROR - Down sample factor is larger than the total number of frames in one recording")

    is_first = True
    output = None
    for each_frame in data:
        output_array = np.expand_dims(each_frame[:,center_of_bright_line], axis=1)

        for i in range(center_of_bright_line+downsample_factor, data.shape[2], downsample_factor):
            to_concat = np.expand_dims(each_frame[:,i], axis=1)
            output_array = np.concatenate((output_array, to_concat), axis=1)
        
        for i in range(center_of_bright_line-downsample_factor, -1, downsample_factor*-1):
            to_concat = np.expand_dims(each_frame[:,i], axis=1)
            output_array = np.insert(output_array, [0], to_concat, axis=1)

        if is_first:
            output = np.expand_dims(output_array, axis=0)
            is_first = False
        else:
            to_concat = np.expand_dims(output_array, axis=0)
            output = np.concatenate((output, to_concat), axis=0)
    return np.array([output, label])
    
