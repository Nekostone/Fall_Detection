import numpy as np


def range_profile(frame_gif_sequence: np.array) -> np.array: # get range profile of each frame in sequence by summing across all velocity bins
    return np.array([np.sum(i, axis=0) for i in frame_gif_sequence])

def dopp_profile(frame_gif_sequence: np.array) -> np.array: # get doppler profile of each frame in sequence by summing across all range bins
    return np.array([np.sum(i, axis=1) for i in frame_gif_sequence])

def range_delta(range_profile_output: np.array) -> np.array: # takes in output of range_profile
    return np.array([np.sum(range_profile_output, axis=1)])

def dopp_delta(dopp_profile_output: np.array) -> np.array: # takes in output of dopp_profile
    return np.array([np.sum(dopp_profile_output, axis=0)])

def com(dopp_or_range_profile: np.array) -> np.array: #takes in normalised output of doppler or range profiles
    ycoord  = np.array([x/2 for x in dopp_or_range_profile])
    xcoord  = np.array(range(1, len(dopp_or_range_profile[0])+1))       # get x coordinates
    xcoord  = xcoord/np.max(xcoord)                                     # normalise
    mass    = np.sum(dopp_or_range_profile,1)                           # get masses for each sequence in profile
    
    xcom    = [np.multiply(n, xcoord) for n in dopp_or_range_profile]
    xcom    = np.sum(xcom)/mass

    ycom    = np.multiply(ycoord, dopp_or_range_profile)
    ycom    = np.sum(ycom)/mass
    return np.array([xcom, ycom])


def feature_defs(input_array: np.array) -> np.array:
    data = input_array[0]
    label = input_array[1]

    range_profile_output = range_profile(data)
    dopp_profile_output = dopp_profile(data)
    range_delta_output = range_delta(range_profile_output)
    dopp_delta_output = dopp_delta(dopp_profile_output)
    com_output_range = com(range_profile_output)
    com_output_dopp = com(dopp_profile_output)

    mean = np.mean(range_profile_output)
    std = np.std(range_profile_output)
    range_profile_output = (range_profile_output - mean)/std

    mean = np.mean(dopp_profile_output)
    std = np.std(dopp_profile_output)
    dopp_profile_output = (dopp_profile_output - mean)/std

    mean = np.mean(range_delta_output)
    std = np.std(range_delta_output)
    range_delta_output = (range_delta_output - mean)/std

    mean = np.mean(dopp_delta_output)
    std = np.std(dopp_delta_output)
    dopp_delta_output = (dopp_delta_output - mean)/std

    mean = np.mean(com_output_range)
    std = np.std(com_output_range)
    com_output_range = (com_output_range - mean)/std

    mean = np.mean(com_output_dopp)
    std = np.std(com_output_dopp)
    com_output_dopp = (com_output_dopp - mean)/std

    # flatten all
    range_profile_output = range_profile_output.flatten()
    dopp_profile_output = dopp_profile_output.flatten()
    range_delta_output = range_delta_output.flatten()
    dopp_delta_output = dopp_delta_output.flatten()
    com_output_range = com_output_range.flatten()
    com_output_dopp = com_output_dopp.flatten()

    output_data = np.concatenate((range_profile_output, dopp_profile_output), axis=0)
    output_data = np.concatenate((output_data, range_delta_output), axis=0)
    output_data = np.concatenate((output_data, dopp_delta_output), axis=0)
    output_data = np.concatenate((output_data, com_output_range), axis=0)
    output_data = np.concatenate((output_data, com_output_dopp), axis=0)

    return np.array([output_data, label])

