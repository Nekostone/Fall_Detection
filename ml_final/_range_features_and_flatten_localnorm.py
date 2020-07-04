import numpy as np


# Second iteration, try with range_profile, dopp_profile, range_delta, and dopp_delta

def range_profile(frame_gif_sequence: np.ndarray) -> np.ndarray: # get range profile of each frame in sequence by summing across all velocity bins
    return np.array([np.sum(i, axis=1) for i in frame_gif_sequence]) 

def delta_range_profile(frame_gif_sequence: np.ndarray) -> np.ndarray: # takes in 6 frames and returns the delta in range profiles for all frames excluding the first
    range_profiles = range_profile(frame_gif_sequence)
    deltas = np.array([np.subtract(range_profiles[i], range_profiles[i-1]) for i in range(1,5)])

    return deltas # expected output length 512

def dopp_profile(frame_gif_sequence: np.ndarray) -> np.ndarray: # get doppler profile of each frame in sequence by summing across all range bins
    return np.array([np.sum(i, axis=0) for i in frame_gif_sequence]) # expected output length 320

def delta_dopp_profile(frame_gif_sequence: np.ndarray) -> np.ndarray: # takes in 6 frames and returns the delta in range profiles for all frames excluding the first
    dopp_profiles = dopp_profile(frame_gif_sequence)
    deltas = np.array([np.subtract(dopp_profiles[i], dopp_profiles[i-1]) for i in range(1,5)])

    return deltas # expected output length 256

def range_features_and_flatten_localnorm(input_array: np.ndarray) -> np.ndarray:
    data = input_array[0]
    label = input_array[1]

    # retrieve features and flatten separately
    range_output = range_profile(data)
    mean = np.mean(range_output)
    std = np.std(range_output)
    range_output = range_output.flatten()
    if std == 0:
        std = 1
    range_output = (range_output - mean)/std

    delta_range_output = delta_range_profile(data)
    mean = np.mean(delta_range_output)
    std = np.std(delta_range_output)
    delta_range_output = delta_range_output.flatten()
    if std == 0:
        std = 1
    delta_range_output = (delta_range_output - mean)/std

    delta_dopp_output = delta_dopp_profile(data)
    mean = np.mean(delta_dopp_output)
    std = np.std(delta_dopp_output)
    if std == 0:
        std = 1
    delta_dopp_output = delta_dopp_output.flatten()
    delta_dopp_output = (delta_dopp_output - mean)/std

    # print("Delta range:{0}, delta dopp:{1}".format(delta_range_output.shape,delta_dopp_output.shape))

    # concat all features into a 1D array
    output_data = np.concatenate((range_output, delta_range_output, delta_dopp_output), axis=0)

    return np.array([output_data, label])
