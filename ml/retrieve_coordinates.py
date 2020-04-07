"""
Output of retrieve_coordinates is a csv file


TODO
- Create coordinates numpy array based on output
- Find minimum number of coordinates across all frames and across all numpy arrays
- Ensure all coordinates is not more than minimum number
- Resulting coordinates are fixed number of coordinates across all frames and all numpy arrays where after thresholding filter, 
  element value is >0
- Input resulting coordinates into SVM
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


falldata_dir = ""
nonfalldata_dir = ""
current_dir = os.path.dirname(os.path.realpath(__file__))
number_of_cores_available = 6



# generate coordinates numpy array
def retrieve_coordinates(process_id, global_dict, list_of_fall_files, input_fall_folder_dir, list_of_notfall_files, input_notfall_folder_dir, output_dir):
    """
    Run per process. Goes through fixed set of numpy arrays and write non-zero coordinates 
    into a structured nested dict (which is saved as a json to output_dir). Returns dir of temp json file and the minimum number of coordinates found in a frame across all files

    Data format saved in temp
    {'fall': {
        'filename': [
            {Each element in this list represents a frame of radar data in a time series array
                'values_list': [A list of all values in the array],
                'coordinates_dict': {
                    value: [
                        [coordinates in the array with this value],
                    ]
                }
            }
        ]},
    'not_fall': {...}
    }

    Final coordinates saved as .npy file (one file for fall, one for notfall)
    np.array([
        [
            file_name,
            [coordinates in frame 0],
            [coordinates in frame 1],
            ...
        ],
        [...]
    ])
    """
    import numpy as np

    # temp = [[list_of_fall_files, input_fall_folder_dir], [list_of_notfall_files, input_notfall_folder_dir]]
    temp = {"fall": [list_of_fall_files, input_fall_folder_dir], "not_fall": [list_of_notfall_files, input_notfall_folder_dir]}
    total_length = len(list_of_fall_files) + len(list_of_notfall_files)
    min_coordinates_count = float('inf')
    count = 0

    data_across_all_categories = {}
    for each_category in temp:
        data_across_all_file = {}
        for each_file in temp[each_category][0]:
            data_across_all_frame = []
            input_dir = os.path.join(temp[each_category][1], each_file)
            print("Process {0}: processing file {1}/{2}; input_dir: {3}".format(process_id, count, total_length, input_dir))

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
            data_across_all_file[each_file] = data_across_all_frame
        data_across_all_categories[each_category] = data_across_all_file
    # print("data_across_all_categories: {0}".format(data_across_all_categories))
    # print("min_coordinates_count: {0}".format(min_coordinates_count))

    with open(output_dir, "w+") as writefile:
        json.dump(data_across_all_categories, writefile)

    global_dict[process_id] = min_coordinates_count


def minimum_coordinates(process_id, input_dir, min_coordinates_count):
    """
    Ensure that each frame would not have more than min_coodinates_count of coordinates inside. We'll keep coordinates that point to the largest value.
    """
    with open(input_dir) as readfile:
        raw_data = json.load(readfile)
        data = copy.deepcopy(raw_data)


        for each_category in data:
            all_files_in_category = data[each_category]
            for each_file in all_files_in_category:
                all_frames_in_file = all_files_in_category[each_file]
                for each_frame in all_frames_in_file:

                    # sort values_list (ascending)
                    values_list = each_frame["values_list"]
                    values_list_length = len(values_list)
                    values_list.sort()

                    # remove random coordinates that point to the value that needs to be deleted
                    coordinates_dict = each_frame["coordinates_dict"]
                    values_to_remove = values_list[:values_list_length-min_coordinates_count]
                    for each_value_to_remove in values_to_remove:
                        coordinate_to_remove = random.choice(coordinates_dict[str(each_value_to_remove)])
                        coordinates_dict[str(each_value_to_remove)].remove(coordinate_to_remove)

                        # if value no longer have any coordinates, delete the value
                        if len(coordinates_dict[str(each_value_to_remove)]) == 0:
                            del coordinates_dict[str(each_value_to_remove)]
                    
                    # remove values_list (we no longer need it)
                    del each_frame["values_list"]

    
    with open(input_dir, "w") as writefile:
        json.dump(data, writefile)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_fall_dir', required=True, type = str, dest='input_fall_dir', help="Directory of input fall data")
    parser.add_argument('--input_notfall_dir', required=True, type = str, dest='input_notfall_dir', help="Directory of input nonfall data")
    parser.add_argument('--output_fall_dir', required=True, type = str, dest='output_fall_dir', help="Directory of json outputfile that stores final processed coordinates of fall data")
    parser.add_argument('--output_notfall_dir', required=True, type = str, dest='output_notfall_dir', help="Directory of json outputfile that stores final processed coordinates of not fall data")
    parser.add_argument('--number_of_parallel_processes', required=True, type = int, dest='number_of_parallel_processes', help="Number of parallel processes to use (in doubt just put 1)")
    args = parser.parse_args()

    # create temp folder to store metadata (return error if folder alr exist)
    temp_folder_dir = os.path.join(current_dir, "temp")
    if os.path.exists(temp_folder_dir):
        print("ERROR - Temp folder already exist. Please delete before running script")
        exit(0)
    os.mkdir(temp_folder_dir)

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
            output_dir = os.path.join(temp_folder_dir, "{0}_coordinates_dict.json".format(i))
            output_file_list.append(output_dir)

            p = multiprocessing.Process(target=retrieve_coordinates, args=(i, global_dict, data[0][i], args.input_fall_dir, data[1][i], args.input_notfall_dir, output_dir))
            p.start()
            process_list.append(p)

        for i in range(args.number_of_parallel_processes):
            process_list[i].join()
    
        for i in global_dict:
            global_minimum_coordinates_count = min(global_minimum_coordinates_count, global_dict[i])
        
    # ensure that all json files have only the minimum number of coordinates across all files (i.e. same number of coordinates)
    process_list = []
    for i in range(args.number_of_parallel_processes):
        p = multiprocessing.Process(target=minimum_coordinates, args=(i, output_file_list[i], global_minimum_coordinates_count))
        p.start()
        process_list.append(p)

    for i in range(args.number_of_parallel_processes):
        process_list[i].join()
    
    # combine all json files into one final json file
    for each_category in ["fall", "not_fall"]:
        if each_category == "fall":
            file_to_write = args.output_fall_dir
        elif each_category == "not_fall":
            file_to_write = args.output_notfall_dir

        all_data_in_category = []
        for each_file in output_file_list:
            with open(each_file) as readfile:
                data = json.load(readfile)
                # print("data: ", data)

                all_data_in_each_metadata = []
                all_files_in_category = data[each_category]
                for each_file in all_files_in_category:
                    all_frames_in_category = all_files_in_category[each_file]
                    # all_frames_data = [each_file]
                    all_frames_data = [each_file]
                    for each_frame in all_frames_in_category:
                        all_values_in_frame = each_frame["coordinates_dict"]
                        each_frame_data = []
                        for each_value in all_values_in_frame:
                            all_coordinates_for_value = all_values_in_frame[each_value]
                            each_frame_data += all_coordinates_for_value
                        # print("each_frame_data: {0}".format(each_frame_data))
                        all_frames_data.append(each_frame_data)
                    # print("all_frames_data: {0}".format(all_frames_data))
                    all_data_in_each_metadata.append(all_frames_data)
                # print("all_data_in_each_metadata: ", all_data_in_each_metadata)
            all_data_in_category += all_data_in_each_metadata
        # print("all_data_in_category: ", all_data_in_category)

        all_data_in_category = np.array(all_data_in_category)
        np.save(file_to_write, all_data_in_category)

                

                
    # cleanup
    subprocess.run("rm -rf {0}".format(temp_folder_dir), shell=True)