import numpy as np
# frame_gif_sequence assumes the form of a radar-data cube:
# 0th dimension: Range
# 1st dimension: Doppler
# 2nd dimension: Time

# First iteration, try with frame_energy, delta_energy, and energy_temporal_derivative

def frame_energy(frame_gif_sequence: np.ndarray) -> np.ndarray: # sum values across each in Radar Data Cube and output 1D array
    return np.array([np.sum(i) for i in frame_gif_sequence])

def delta_energy(frame_energy: np.ndarray) -> float: # output scalar value of changes across frame profile. Takes frame_energy output as input
    diff = np.array(frame_energy[0:-1])
    diff -= np.array(frame_energy[1:])
    diff = np.sum(diff)
    return np.expand_dims(diff, axis=0)

def energy_temporal_derivative(frame_energy: np.ndarray) -> np.ndarray: # output vector value of changes across frame profile. Takes frame_energy output as input
    diff = np.array(frame_energy[0:-1])
    diff -= np.array(frame_energy[1:])
    return diff

def energy_features_and_flatten(input_array: np.ndarray) -> np.ndarray:
    data = input_array[0]
    label = input_array[1]

    # retrieve features separately
    frame_energy_output = frame_energy(data)
    delta_energy_output = delta_energy(frame_energy_output)
    energy_temporal_derivative_output = energy_temporal_derivative(frame_energy_output)

    # concat all as features for one recording
    output_data = np.concatenate((frame_energy_output, delta_energy_output), axis=0)
    output_data = np.concatenate((output_data, energy_temporal_derivative_output), axis=0)

    return np.array([output_data, label])
