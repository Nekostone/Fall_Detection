"""
Output of retrieve_coordinates is a csv file

"""


import numpy as np
import os
import json
import argparse
import random
import subprocess
import multiprocessing
import copy
import csv


# generate coordinates numpy array
def retrieve_coordinates(process_id, global_dict, list_of_fall_files, input_fall_folder_dir, list_of_notfall_files, input_notfall_folder_dir, output_fall_folder_dir, output_notfall_folder_dir):
    """
    Run per process. Goes through fixed set of numpy arrays and write non-zero coordinates 
    into a structured nested dict (which is saved as a json to output_dir). Returns dir of temp json file and the minimum number of coordinates found in a frame across all files

    Output (for each file)
    [
        {Each element in this list represents a frame of radar data in a time series array
            'values_list': [A list of all values in the array],
            'coordinates_dict': {
                value: [
                    [coordinates in the array with this value],
                ]
            }
        }
    ]
    """
    # temp = [[list_of_fall_files, input_fall_folder_dir], [list_of_notfall_files, input_notfall_folder_dir]]
    temp = {"fall": [list_of_fall_files, input_fall_folder_dir], "not_fall": [list_of_notfall_files, input_notfall_folder_dir]}
    total_length = len(list_of_fall_files) + len(list_of_notfall_files)
    min_coordinates_count = float('inf')
    count = 0

    for each_category in temp:
        for each_file in temp[each_category][0]:
            data_across_all_frame = []
            input_dir = os.path.join(temp[each_category][1], each_file)
            # print("Process {0}: processing file {1}/{2}; input_dir: {3}".format(process_id, count, total_length, input_dir))

            input_data = np.load(input_dir)
            # for each frame
            for i in range(input_data.shape[0]):
                values_list = []
                coordinates_dict = {}
                input_frame = input_data[i]

                coordinates_count = 0
                for y in range(len(input_frame)):
                    for x in range(len(input_frame[0])):
                        val = float(input_frame[y][x])
                        if val != 0:
                            # if element in a frame is not zero, save its value into values_list and coordinates into coordinates_dict
                            values_list.append(val)
                            coordinates_dict[val] = coordinates_dict.get(val, [])
                            coordinates_dict[val].append([x,y])

                            # count number of such element in the frame
                            coordinates_count += 1
                data_across_all_frame.append({"values_list": values_list, "coordinates_dict": coordinates_dict})

                min_coordinates_count = min(min_coordinates_count, coordinates_count)
            count += 1

            if each_category == "fall":
                output_parent_dir = output_fall_folder_dir
            elif each_category == "not_fall":
                output_parent_dir = output_notfall_folder_dir

            output_dir = os.path.join(output_parent_dir, each_file)
            np.save(output_dir, np.array(data_across_all_frame))

    global_dict[process_id] = min_coordinates_count


def minimum_coordinates(process_id, min_coordinates_count, list_of_fall_files, input_fall_folder_dir, list_of_notfall_files, input_notfall_folder_dir):
    """
    Ensure that each frame would not have more than min_coodinates_count of coordinates inside. We'll keep coordinates that point to the largest value.

    output format (each file would look like this):
    [
        each frame
        {
            each value: [
                [coordinates of this particular value],
                ...
            ]
        },
        ...
    ]
    """
    temp = {"fall": [list_of_fall_files, input_fall_folder_dir], "not_fall": [list_of_notfall_files, input_notfall_folder_dir]}

    for each_category in temp:
        for each_file in temp[each_category][0]:
            input_dir = os.path.join(temp[each_category][1], each_file)
            all_frames_in_file = np.load(input_dir, allow_pickle=True)

            file_retained_values = []  # @@@ temp logging

            for each_frame in all_frames_in_file:
                # sort values_list (ascending)
                values_list = each_frame["values_list"]
                values_list_length = len(values_list)
                values_list.sort()
                # remove random coordinates that point to the value that needs to be deleted
                coordinates_dict = each_frame["coordinates_dict"]

                values_to_remove = values_list[:values_list_length-min_coordinates_count]
                for each_value_to_remove in values_to_remove:
                    coordinate_to_remove = random.choice(coordinates_dict[each_value_to_remove])
                    coordinates_dict[each_value_to_remove].remove(coordinate_to_remove)

                    # if value no longer have any coordinates, delete the value
                    if len(coordinates_dict[each_value_to_remove]) == 0:
                        del coordinates_dict[each_value_to_remove]

                file_retained_values.append(list(coordinates_dict.keys()))  # @@@ temp logging

                # remove values_list (we no longer need it)
                del each_frame["values_list"]

            logging_dir = os.path.join("/home/xubuntu/Desktop/temp/", each_file)  # @@@ temp logging
            np.save(logging_dir, np.array(file_retained_values))  # @@@ temp logging

            np.save(input_dir, all_frames_in_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_fall_dir', required=True, type = str, dest='input_fall_dir', help="Directory of input fall data")
    parser.add_argument('--input_notfall_dir', required=True, type = str, dest='input_notfall_dir', help="Directory of input nonfall data")
    parser.add_argument('--output_dir', required=True, type = str, dest='output_dir', help="Directory of npy outputfile that stores final processed coordinates of fall data")
    parser.add_argument('--max_coordinates', default=1000, required=True, type = int, dest='max_coordinates', help="Sets the maximum number of coordinates permitted per frame in a file. If the global minumum is lower than this number, that will be used instead.")
    parser.add_argument('--number_of_parallel_processes', required=True, type = int, dest='number_of_parallel_processes', help="Number of parallel processes to use (in doubt just put 1)")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.realpath(__file__))

    temp_folder_dir = os.path.join(args.output_dir, "retrieve_coordinates_{0}coordinates".format(args.max_coordinates))
    if os.path.exists(temp_folder_dir):
        subprocess.call("rm -rf {0}".format(temp_folder_dir), shell=True)
        print("ERROR - folder {0} already exist. Deleting old folder...".format(temp_folder_dir))
    os.mkdir(temp_folder_dir)

    retrievecoords_fall_outputdir = os.path.join(temp_folder_dir, "fall/")
    if os.path.exists(retrievecoords_fall_outputdir):
        subprocess.call("rm -rf {0}".format(retrievecoords_fall_outputdir), shell=True)
        print("ERROR - folder {0} already exist. Deleting old folder...".format(retrievecoords_fall_outputdir))
    os.mkdir(retrievecoords_fall_outputdir)

    retrievecoords_notfall_outputdir = os.path.join(temp_folder_dir, "not_fall/")
    if os.path.exists(retrievecoords_notfall_outputdir):
        subprocess.call("rm -rf {0}".format(retrievecoords_notfall_outputdir), shell=True)
        print("ERROR - folder {0} already exist. Deleting old folder...".format(retrievecoords_notfall_outputdir))
    os.mkdir(retrievecoords_notfall_outputdir)


    # START - multithreaded implementation
    # split files evenly to be processed by each process
    data = []
    for i in [args.input_fall_dir, args.input_notfall_dir]:
        list_of_files = os.listdir(i)
        d = len(list_of_files)//args.number_of_parallel_processes
        sublist_of_files = []

        i = -1
        for i in range(args.number_of_parallel_processes-1):
            sublist_of_files.append(list_of_files[i*d:(i+1)*d])
        sublist_of_files.append(list_of_files[(i+1)*d:])

        data.append(sublist_of_files)

    # retrieve coordinates from npy files, and return the minimum number of coordinates in a frame across all files
    global_minimum_coordinates_count = float("inf")
    output_file_list = []
    with multiprocessing.Manager() as manager:
        global_dict = manager.dict()  # used for processes to send the minimum number of coordinates in a frame to the master process
        process_list = []
        for i in range(args.number_of_parallel_processes):
            p = multiprocessing.Process(target=retrieve_coordinates, args=(i, global_dict, data[0][i], args.input_fall_dir, data[1][i], args.input_notfall_dir, retrievecoords_fall_outputdir, retrievecoords_notfall_outputdir))
            p.start()
            process_list.append(p)

        for i in range(args.number_of_parallel_processes):
            process_list[i].join()
    
        for i in global_dict:
            global_minimum_coordinates_count = min(global_minimum_coordinates_count, global_dict[i])

    global_minimum_coordinates_count = min(global_minimum_coordinates_count, args.max_coordinates)

    output_file_list = []
    process_list = []
    for i in range(args.number_of_parallel_processes):
        p = multiprocessing.Process(target=minimum_coordinates, args=(i, global_minimum_coordinates_count, data[0][i], retrievecoords_fall_outputdir, data[1][i], retrievecoords_notfall_outputdir))
        p.start()
        process_list.append(p)

    for i in range(args.number_of_parallel_processes):
        process_list[i].join()
