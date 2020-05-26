import numpy as np
# frame_gif_sequence assumes the form of a radar-data cube:
# 0th dimension: Range
# 1st dimension: Doppler
# 2nd dimension: Time

# Second iteration, try with range_profile, dopp_profile, range_delta, and dopp_delta

def range_profile(frame_gif_sequence: np.ndarray) -> np.ndarray: # get range profile of each frame in sequence by summing across all velocity bins
    return np.array([np.sum(i, axis=1) for i in frame_gif_sequence])

def dopp_profile(frame_gif_sequence: np.ndarray) -> np.ndarray: # get doppler profile of each frame in sequence by summing across all range bins
    return np.array([np.sum(i, axis=0) for i in frame_gif_sequence])

def range_delta(range_profile_output: np.ndarray) -> np.ndarray: # takes in output of range_profile
    return np.array([np.sum(range_profile_output, axis=0)])

def dopp_delta(dopp_profile_output: np.ndarray) -> np.ndarray: # takes in output of range_profile
    return np.array([np.sum(dopp_profile_output, axis=0)])

def range_features_and_flatten(input_array: np.ndarray) -> np.ndarray:
    data = input_array[0]
    label = input_array[1]

    # retrieve features and flatten separately
    range_profile_output = range_profile(data).flatten()
    # range_delta_output = range_delta(range_profile_output).flatten()
    range_delta_output = range_delta(range_profile_output)
    dopp_profile_output = dopp_profile(data).flatten()
    # dopp_delta_output = dopp_delta(dopp_profile_output).flatten()
    dopp_delta_output = dopp_delta(dopp_profile_output)

    # concat all features into a 1D array
    output_data = np.concatenate((range_profile_output, range_delta_output), axis=0)
    output_data = np.concatenate((output_data, dopp_profile_output), axis=0)
    output_data = np.concatenate((output_data, dopp_delta_output), axis=0)

    return np.array([output_data, label])
