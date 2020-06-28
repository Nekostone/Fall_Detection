import numpy as np


# Second iteration, try with range_profile, dopp_profile, range_delta, and dopp_delta

def range_profile(frame_gif_sequence: np.ndarray) -> np.ndarray: # get range profile of each frame in sequence by summing across all velocity bins
    return np.array([np.sum(i, axis=0) for i in frame_gif_sequence])

def dopp_profile(frame_gif_sequence: np.ndarray) -> np.ndarray: # get doppler profile of each frame in sequence by summing across all range bins
    return np.array([np.sum(i, axis=1) for i in frame_gif_sequence])

def time_integrate(arr: "(5, rownum, colnum) nparray") -> "(rownum, colnum) nparray":
    time_dept_rdm = arr[0]
    for frame in range(1,5):
        time_dept_rdm = np.maximum(time_dept_rdm, arr[frame] )

    return time_dept_rdm

def range_features_and_flatten_localnorm(input_array: np.ndarray) -> np.ndarray:
    data = input_array[0]
    label = input_array[1]

    time_integrate_output = time_integrate(data)
    time_integrate_output = np.expand_dims(time_integrate_output, axis=0)

    # retrieve features and flatten separately
    range_profile_output = range_profile(time_integrate_output)
    range_profile_output = range_profile(data)
    mean = np.mean(range_profile_output)
    std = np.std(range_profile_output)
    range_profile_output = range_profile_output.flatten()
    range_profile_output = (range_profile_output - mean)/std

    dopp_profile_output = dopp_profile(time_integrate_output)
    dopp_profile_output = dopp_profile(data)
    mean = np.mean(dopp_profile_output)
    std = np.std(dopp_profile_output)
    dopp_profile_output = dopp_profile_output.flatten()
    dopp_profile_output = (dopp_profile_output - mean)/std

    # concat all features into a 1D array
    output_data = np.concatenate((range_profile_output, dopp_profile_output), axis=0)

    return np.array([output_data, label])
