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

"""
def dopp_delta(dopp_profile_output: np.ndarray) -> np.ndarray: # takes in output of range_profile
    return np.array([np.sum(dopp_profile_output, axis=0)])
"""

def dopp_delta(dopp_profile_output: np.ndarray) -> np.ndarray: # takes in output of dopp_profile
    return np.array([np.sum(dopp_profile_output, axis=1)])



def range_features_normalized(train_x, test_x):
    # retrieve range profile and standardise to train_x data
    range_profile_output_train_x = []
    for each_recording in train_x:
        range_profile_output_train_x.append(range_profile(each_recording))
    range_profile_output_test_x = []
    for each_recording in test_x:
        range_profile_output_test_x.append(range_profile(each_recording))

    mean = np.mean(range_profile_output_train_x)
    std = np.std(range_profile_output_train_x)

    for each_recording in range_profile_output_train_x:
        each_recording = (each_recording-mean)/std
    for each_recording in range_profile_output_test_x:
        each_recording = (each_recording-mean)/std

    # retrieve range_delta features
    range_delta_output_train_x = []
    for each_recording in range_profile_output_train_x:
        range_delta_output_train_x.append(range_delta(each_recording))
    range_delta_output_test_x = []
    for each_recording in range_profile_output_test_x:
        range_delta_output_test_x.append(range_delta(each_recording))

    # retrieve dopp profile and standardise to train_x data
    dopp_profile_output_train_x = []
    for each_recording in train_x:
        dopp_profile_output_train_x.append(dopp_profile(each_recording))
    dopp_profile_output_test_x = []
    for each_recording in test_x:
        dopp_profile_output_test_x.append(dopp_profile(each_recording))

    mean = np.mean(dopp_profile_output_train_x)
    std = np.std(dopp_profile_output_train_x)

    for each_recording in dopp_profile_output_train_x:
        each_recording = (each_recording-mean)/std
    for each_recording in dopp_profile_output_test_x:
        each_recording = (each_recording-mean)/std

    # retrieve dopp_delta features
    dopp_delta_output_train_x = []
    for each_recording in dopp_profile_output_train_x:
        dopp_delta_output_train_x.append(dopp_delta(each_recording))
    dopp_delta_output_test_x = []
    for each_recording in dopp_profile_output_test_x:
        dopp_delta_output_test_x.append(dopp_delta(each_recording))

    # flatten and concat features for each recording
    train_x_output = []
    for i in range(len(train_x)):
        range_profile_output_train_x[i] = range_profile_output_train_x[i].flatten()
        train_x_output.append(range_profile_output_train_x[i])

        range_delta_output_train_x[i] = range_delta_output_train_x[i].flatten()
        train_x_output[i] = np.concatenate((train_x_output[i], range_delta_output_train_x[i]))

        dopp_profile_output_train_x[i] = dopp_profile_output_train_x[i].flatten()
        train_x_output[i] = np.concatenate((train_x_output[i], dopp_profile_output_train_x[i]))

        dopp_delta_output_train_x[i] = dopp_delta_output_train_x[i].flatten()
        train_x_output[i] = np.concatenate((train_x_output[i], dopp_delta_output_train_x[i]))

    test_x_output = []
    for i in range(len(test_x)):
        range_profile_output_test_x[i] = range_profile_output_test_x[i].flatten()
        test_x_output.append(range_profile_output_test_x[i])

        range_delta_output_test_x[i] = range_delta_output_test_x[i].flatten()
        test_x_output[i] = np.concatenate((test_x_output[i], range_delta_output_test_x[i]))

        dopp_profile_output_test_x[i] = dopp_profile_output_test_x[i].flatten()
        test_x_output[i] = np.concatenate((test_x_output[i], dopp_profile_output_test_x[i]))

        dopp_delta_output_test_x[i] = dopp_delta_output_test_x[i].flatten()
        test_x_output[i] = np.concatenate((test_x_output[i], dopp_delta_output_test_x[i]))

    return train_x_output, test_x_output
