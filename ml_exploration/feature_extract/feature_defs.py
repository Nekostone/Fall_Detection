import numpy as np
import itertools as it
# frame_gif_sequence assumes the form of a radar-data cube:
# 0th dimension: Range
# 1st dimension: Doppler
# 2nd dimension: Time

# First iteration, try with frame_energy, delta_energy, and energy_temporal_derivative

def frame_energy(frame_gif_sequence: np.array) -> np.array: # sum values across each in Radar Data Cube and output 1D array
    return np.array([np.sum(i) for i in frame_gif_sequence])

def total_energy(frame_gif_sequence: np.array) -> float:
    return np.sum(frame_gif_sequence)

def delta_energy(frame_energy: np.array) -> float: # output scalar value of changes across frame profile. Takes frame_energy output as input
    diff = frame_energy[1:]
    new = frame_energy[0:-1] - diff
    return np.sum(new)

def energy_temporal_derivative(frame_energy: np.array) -> np.array: # output vector value of changes across frame profile. Takes frame_energy output as input
    diff = frame_energy[1:]
    new = frame_energy[0:-1] - diff
    return new

def first_iteration(input_array: np.array) -> np.array:
    """
    No local normalise
    true_positive: 571, true_negative: 46, false_positive: 530, false_negative: 17
    true_positive: 581, true_negative: 28, false_positive: 548, false_negative: 7
    """


    data = input_array[0]
    label = input_array[1]

    frame_energy_output = frame_energy(data)
    total_energy_output = np.expand_dims(total_energy(data), axis=0)
    delta_energy_output = np.expand_dims(delta_energy(frame_energy_output), axis=0)
    energy_temporal_derivative_output = energy_temporal_derivative(frame_energy_output).flatten()

    # flatten all
    frame_energy_output = frame_energy_output.flatten()

    output_data = np.concatenate((frame_energy_output, total_energy_output), axis=0)
    output_data = np.concatenate((output_data, delta_energy_output), axis=0)
    output_data = np.concatenate((output_data, energy_temporal_derivative_output), axis=0)

    """
    print("frame_energy_output shape: {0}".format(frame_energy_output.shape))
    print("total_energy_output shape: {0}".format(total_energy_output.shape))
    print("delta_energy_output shape: {0}".format(delta_energy_output.shape))
    print("energy_temporal_derivative_output shape: {0}".format(energy_temporal_derivative_output.shape))
    """

    return np.array([output_data, label])
    

# Second iteration, try with range_profile, dopp_profile, range_delta, and dopp_delta

def range_profile(frame_gif_sequence: np.array) -> np.array: # get range profile of each frame in sequence by summing across all velocity bins
    return np.array([np.sum(i, axis=1) for i in frame_gif_sequence])

def dopp_profile(frame_gif_sequence: np.array) -> np.array: # get doppler profile of each frame in sequence by summing across all range bins
    return np.array([np.sum(i, axis=0) for i in frame_gif_sequence])

def range_delta(range_profile_output: np.array) -> np.array: # takes in output of range_profile
    return np.array([np.sum(range_profile_output, axis=0)])

def dopp_delta(dopp_profile_output: np.array) -> np.array: # takes in output of dopp_profile
    return np.array([np.sum(dopp_profile_output, axis=1)])

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


def second_iteration(input_array: np.array) -> np.array:
    data = input_array[0]
    label = input_array[1]

    range_profile_output = range_profile(data)
    dopp_profile_output = dopp_profile(data)
    range_delta_output = range_delta(range_profile_output)
    dopp_delta_output = dopp_delta(dopp_profile_output)
    com_output_range = com(range_profile_output)
    com_output_dopp = com(dopp_profile_output)
    # print("com_output_range: {0}".format(com_output_range))
    # print("com_output_dopp: {0}".format(com_output_dopp))
    print(range_profile_output.shape)

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

