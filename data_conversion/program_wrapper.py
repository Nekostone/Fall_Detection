"""
Modes:
1 - write_to_npy
2 - npy_to_gif

"""


import os
import argparse
from multiprocessing import Process

##Linux dir##
# program_abs_dir = "/home/xubuntu/Desktop/Fall_Detection/write_to_npy.py"
# input_fall_dir = "/home/xubuntu/Desktop/sensor_data/raw/fall/"
# input_not_fall_dir = "/home/xubuntu/Desktop/sensor_data/raw/not_fall/"
# output_fall_dir = "/home/xubuntu/Desktop/sensor_data/processed/fall/"
# output_not_fall_dir = "/home/xubuntu/Desktop/sensor_data/processed/not_fall/"

##Windows dir
program_abs_dir = "C:\\Users\\user\\Documents\\Capstone\\SourceCode\\Fall_Detection\\data_conversion\\write_to_npy.py"
input_fall_dir = "C:\\Users\\user\\Documents\\Capstone\\SourceCode\\Fall_Data\\radar_sensor_data\\fall"
input_not_fall_dir = "C:\\Users\\user\\Documents\\Capstone\\SourceCode\\Fall_Data\\radar_sensor_data\\not_fall"
output_fall_dir = "C:\\Users\\user\\Documents\\Capstone\\SourceCode\\Fall_Data\\radar_sensor_data\\processed\\fall"
output_not_fall_dir = "C:\\Users\\user\\Documents\\Capstone\\SourceCode\\Fall_Data\\radar_sensor_data\\processed\\not_fall"

number_of_parallel_processes = 4

def write_to_npy(process_id, list_of_fall_files, input_fall_folder_dir, output_fall_folder_dir, list_of_notfall_files, input_notfall_folder_dir, output_notfall_folder_dir):
    import subprocess

    temp = [[list_of_fall_files, input_fall_folder_dir, output_fall_folder_dir], [list_of_notfall_files, input_notfall_folder_dir, output_notfall_folder_dir]]
    total_length = len(list_of_fall_files) + len(list_of_notfall_files)
    count = 0

    for each_category in temp:
        for each_file in each_category[0]:
            input_dir = each_category[1]
            output_dir = each_category[2]
            print("Process {0}: processing file {1}/{2}; input_dir: {3}".format(process_id, count, total_length, input_dir))
            subprocess.call(["python", program_abs_dir, "--input_folder", input_dir,"--input_filename",each_file, "--output_folder", output_dir])
            count += 1

def npy_to_gif(process_id, list_of_fall_files, input_fall_folder_dir, output_fall_folder_dir, list_of_notfall_files, input_notfall_folder_dir, output_notfall_folder_dir):
    import subprocess

    temp = [[list_of_fall_files, input_fall_folder_dir, output_fall_folder_dir], [list_of_notfall_files, input_notfall_folder_dir, output_notfall_folder_dir]]
    total_length = len(list_of_fall_files) + len(list_of_notfall_files)
    count = 0

    for each_category in temp:
        for each_file in each_category[0]:
            input_dir = each_category[1]
            output_dir = each_category[2]
            print("Process {0}: processing file {1}/{2}; input_dir: {3}".format(process_id, count, total_length, input_dir))
            subprocess.call(["python", program_abs_dir, "--input_folder", input_dir,"--input_filename",each_file, "--output_folder", output_dir])
            count += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data files (using existing functions in this folder) across multiple processes')
    parser.add_argument('--mode', required=True, type = int, dest='mode', help="Set the mode of processing of data files")
    parser.add_argument('--program_abs_dir', required=True, type = str, dest='program_abs_dir', help="")
    parser.add_argument('--input_fall_dir', required=True, type = str, dest='input_fall_dir', help="")
    parser.add_argument('--input_not_fall_dir', required=True, type = str, dest='input_not_fall_dir', help="")
    parser.add_argument('--output_fall_dir', type = str, dest='output_fall_dir', help="")
    parser.add_argument('--output_not_fall_dir', type = str, dest='output_not_fall_dir', help="")
    parser.add_argument('--number_of_parallel_processes', required=True, type = int, dest='number_of_parallel_processes', help="Number of parallel processes to use (in doubt just put 1)")
    args = parser.parse_args()

    mode = args.mode
    input_fall_dir = args.input_fall_dir
    input_not_fall_dir = args.input_not_fall_dir
    number_of_parallel_processes = args.number_of_parallel_processes

    # split files evenly to be processed by each process
    #get all filename from fall data#
    os.chdir(input_fall_dir)
    list_of_fall_files = []
    arr = os.listdir()
    for file in arr:
        file = file.split(".")
        filename = file[0]
        if filename == "desktop":
            continue
        elif filename not in list_of_fall_files:
            list_of_fall_files.append(filename)
    #get all filename from not fall data#
    list_of_notfall_files =[]
    os.chdir(input_not_fall_dir)
    arr = os.listdir()
    for file in arr:
        file = file.split(".")
        filename = file[0]
        if filename == "desktop":
            continue
        elif filename not in list_of_notfall_files:
            list_of_notfall_files.append(filename)

    # split all files evenly for processes to process concurrently
    data = []
    for filelist in [list_of_fall_files,list_of_notfall_files]:
        d = len(filelist)//number_of_parallel_processes
        sublist_of_files =[]

        i = -1
        for i in range(number_of_parallel_processes-1):
            sublist_of_files.append(filelist[i*d:(i+1)*d])
        sublist_of_files.append(filelist[(i+1)*d:])

        data.append(sublist_of_files)
    

    # run parallel processes to process files
    process_list = []
    for i in range(number_of_parallel_processes):
        if mode == 1:
            p = Process(target=write_to_npy, args=(i, data[0][i], input_fall_dir, output_fall_dir, data[1][i], input_not_fall_dir, output_not_fall_dir))
        elif mode == 2:
            p = Process(target=npy_to_gif, args=(i, data[0][i], input_fall_dir, output_fall_dir, data[1][i], input_not_fall_dir, output_not_fall_dir))
        p.start()
        process_list.append(p)
    
    for i in range(number_of_parallel_processes):
        process_list[i].join()

    print("done.")