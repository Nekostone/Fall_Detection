import numpy as np
import os
from multiprocessing import Process


number_of_features_to_track = 5
width_of_rim_for_comparison = 1
width_of_ignored_pixels = 1
number_of_parallel_processes = 6

thresholding_dir = "/home/xubuntu/Desktop/Fall_Detection/ml/thresholding.py"

falldata_dir = "/home/xubuntu/Desktop/capstone/ml/raw_data_npy/fall/"
nonfalldata_dir = "/home/xubuntu/Desktop/capstone/ml/raw_data_npy/not_fall/"
processed_parentdir = "/home/xubuntu/Desktop/capstone/ml/thresholded(comparisonrim_ignoredpixels)/"

list_of_falldata = os.listdir(falldata_dir)
list_of_nonfalldata = os.listdir(nonfalldata_dir)


def processfiles(process_id, list_of_fall_files, input_fall_folder_dir, output_fall_folder_dir, list_of_notfall_files, input_notfall_folder_dir, output_notfall_folder_dir):
    import subprocess

    temp = [[list_of_fall_files, input_fall_folder_dir, output_fall_folder_dir], [list_of_notfall_files, input_notfall_folder_dir, output_notfall_folder_dir]]
    total_length = len(list_of_fall_files) + len(list_of_notfall_files)
    count = 0

    for each_category in temp:
        for each_file in each_category[0]:
            input_dir = os.path.join(each_category[1], each_file)
            output_dir = os.path.join(each_category[2], each_file)
            print("Process {0}: processing file {1}/{2}; input_dir: {3}".format(process_id, count, total_length, input_dir))
            subprocess.run(["python3", thresholding_dir, "--input_filename", input_dir, "--output_filename", output_dir, "--width_of_ignored_pixels", str(width_of_ignored_pixels), "--width_of_rim_for_comparison", str(width_of_rim_for_comparison)])
            # subprocess.run("python3 {0} --input_filename {1} --output_filename {2} --width_of_ignored_pixels {3} --width_of_rim_for_comparison {4}".format(thresholding_dir, input_dir, output_dir, width_of_ignored_pixels, width_of_rim_for_comparison), shell=True)
            count += 1



if __name__ == "__main__":
    # create containing folders if does not exist
    folder_name = "{0}_{1}".format(width_of_ignored_pixels, width_of_rim_for_comparison)
    processed_dir = os.path.join(processed_parentdir, folder_name)
    processed_fall_dir = os.path.join(processed_dir, "fall")
    processed_notfall_dir = os.path.join(processed_dir, "not_fall")
    for each_folder_dir in [processed_dir, processed_fall_dir, processed_notfall_dir]:
        if not os.path.exists(each_folder_dir):
            os.mkdir(each_folder_dir)

    # split files evenly to be processed by each process
    data = []
    for i in [falldata_dir, nonfalldata_dir]:
        list_of_files = os.listdir(i)
        d = len(list_of_files)//number_of_parallel_processes
        sublist_of_files = []

        i = -1
        for i in range(number_of_parallel_processes-1):
            sublist_of_files.append(list_of_files[i*d:(i+1)*d])
        sublist_of_files.append(list_of_files[(i+1)*d:])

        data.append(sublist_of_files)

    # run parallel processes to process files
    process_list = []
    for i in range(number_of_parallel_processes):
        p = Process(target=processfiles, args=(i, data[0][i], falldata_dir, processed_fall_dir, data[1][i], nonfalldata_dir, processed_notfall_dir,))
        p.start()
        process_list.append(p)
    
    for i in range(number_of_parallel_processes):
        process_list[i].join()

print("done")
