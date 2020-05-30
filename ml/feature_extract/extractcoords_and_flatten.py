import numpy as np

def extractcoords_and_flatten(input_array, extracted_per_frame):
    data = input_array[0]
    label = input_array[1]
    output_data = []

    if extracted_per_frame > data.shape[0]*data.shape[1]*data.shape[2]:
        exit("ERROR - extracted_per_frame more than total number of points in input data")

    for i in range(data.shape[0]):
        values_list = []
        coordinates_dict = {}
        input_frame = data[i]

        for y in range(data.shape[1]):
            for x in range(data.shape[2]):
                val = float(input_frame[y][x])
                if val != 0:
                    # if element in a frame is not zero, save its value into values_list and coordinates into coordinates_dict
                    values_list.append(val)
                    coordinates_dict[val] = coordinates_dict.get(val, [])
                    coordinates_dict[val].append([y,x])

        values_list.sort(reverse=True)

        for j in range(extracted_per_frame):
            # add coordinate values to output_data list
            value_to_extract = values_list[0]
            coordinates = coordinates_dict[value_to_extract].pop()
            output_data.extend(coordinates)

            del values_list[0]

    return np.array([output_data, label])

