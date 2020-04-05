import argparse
import numpy as np
import os
from multiprocessing import Process


def processfiles(process_id, program_abs_dir, list_of_fall_files, input_fall_folder_dir, output_fall_folder_dir, list_of_notfall_files, input_notfall_folder_dir, output_notfall_folder_dir):
    import subprocess

    temp = [[list_of_fall_files, input_fall_folder_dir, output_fall_folder_dir], [list_of_notfall_files, input_notfall_folder_dir, output_notfall_folder_dir]]
    total_length = len(list_of_fall_files) + len(list_of_notfall_files)
    count = 0

    for each_category in temp:
        for each_file in each_category[0]:
            input_dir = os.path.join(each_category[1], each_file)
            output_dir = os.path.join(each_category[2], each_file)
            print("Process {0}: processing file {1}/{2}; input_dir: {3}".format(process_id, count, total_length, input_dir))
            subprocess.run(["python3", program_abs_dir, "--input_filename", input_dir, "--output_filename", output_dir, "--width_of_ignored_pixels", str(args.width_of_ignored_pixels), "--width_of_rim_for_comparison", str(args.width_of_rim_for_comparison)])
            # subprocess.run("python3 {0} --input_filename {1} --output_filename {2} --width_of_ignored_pixels {3} --width_of_rim_for_comparison {4}".format(program_abs_dir, input_dir, output_dir, args.width_of_ignored_pixels, args.width_of_rim_for_comparison), shell=True)
            count += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply thresholding filter across multiple files in parallel.')
    parser.add_argument('--program_abs_dir', required=True, type = str, dest='program_abs_dir', help="Directory of thresholding.py")
    parser.add_argument('--input_fall_dir', required=True, type = str, dest='input_fall_dir', help="Directory of input fall data")
    parser.add_argument('--input_not_fall_dir', required=True, type = str, dest='input_not_fall_dir', help="Directory of input nonfall data")
    parser.add_argument('--output_dir', type = str, dest='output_dir', help="Parent directory for output for all data")
    parser.add_argument('--width_of_rim_for_comparison', required=True, type = str, dest='width_of_rim_for_comparison', help="Width of the rim of pixels that will be compared with the center pixel in thresholding filter")
    parser.add_argument('--width_of_ignored_pixels', required=True, type = str, dest='width_of_ignored_pixels', help="Wid of the rim of pixels that will be ignored in thresholding filter")
    parser.add_argument('--number_of_parallel_processes', required=True, type = int, dest='number_of_parallel_processes', help="Number of parallel processes to use (in doubt just put 1)")
    args = parser.parse_args()

    # create containing folders if does not exist
    folder_name = "{0}_{1}".format(args.width_of_ignored_pixels, args.width_of_rim_for_comparison)
    processed_dir = os.path.join(args.output_dir, folder_name)
    processed_fall_dir = os.path.join(processed_dir, "fall")
    processed_notfall_dir = os.path.join(processed_dir, "not_fall")
    for each_folder_dir in [processed_dir, processed_fall_dir, processed_notfall_dir]:
        if not os.path.exists(each_folder_dir):
            os.mkdir(each_folder_dir)

    # split files evenly to be processed by each process
    data = []
    for i in [args.input_fall_dir, args.input_not_fall_dir]:
        list_of_files = os.listdir(i)
        d = len(list_of_files)//args.number_of_parallel_processes
        sublist_of_files = []

        i = -1
        for i in range(args.number_of_parallel_processes-1):
            sublist_of_files.append(list_of_files[i*d:(i+1)*d])
        sublist_of_files.append(list_of_files[(i+1)*d:])

        data.append(sublist_of_files)

    # run parallel processes to process files
    process_list = []
    for i in range(args.number_of_parallel_processes):
        p = Process(target=processfiles, args=(i, args.program_abs_dir, data[0][i], args.input_fall_dir, processed_fall_dir, data[1][i], args.input_not_fall_dir, processed_notfall_dir,))
        p.start()
        process_list.append(p)
    
    for i in range(args.number_of_parallel_processes):
        process_list[i].join()

print("done")
