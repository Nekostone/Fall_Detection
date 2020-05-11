"""
Iterates the training of multiple SVM models based on various possible values for threshold filter parameters.
"""
import subprocess
import numpy as np
import os
import csv



raw_npy_dir_fall = "/home/xubuntu/Desktop/sensor_data/raw_but_npy/fall/raw"
raw_npy_dir_notfall = "/home/xubuntu/Desktop/sensor_data/raw_but_npy/not_fall/raw"
number_of_parallel_processes = 11
overall_output = "/home/xubuntu/Desktop/Fall_Detection/ml/svm_iterator_results.csv"


# first-time creation of overall_output
if os.path.isfile(overall_output):
    subprocess.call("rm -rf {0}".format(overall_output), shell=True)
with open(overall_output, 'w+') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["width of rim for comparison", "width of ignored pixels", "max coordinates", "true positive", "true negative", "false positive", "false negative"])


# check shape of first datafile in raw_npy_dir_fall
fall_file_list = os.listdir(raw_npy_dir_fall)
first_file_dir = os.path.join(raw_npy_dir_fall, fall_file_list[0])
first_file_shape = np.load(first_file_dir).shape

max_length_of_threshold_filter = min(first_file_shape[1:])//4
max_width = (max_length_of_threshold_filter-1)//2

print("max_width: {0}".format(max_width))

for submax_width in range(max_width, 1, -1):
    for i in range(1, submax_width-1):
        width_of_rim_for_comparison_0 = i
        width_of_ignored_pixels_0 = submax_width-i

        print("{0}_{1}: Start".format(width_of_rim_for_comparison_0, width_of_ignored_pixels_0))

        program_abs_dir_0 = "/home/xubuntu/Desktop/Fall_Detection/ml/thresholding.py"
        input_fall_dir_0 = raw_npy_dir_fall
        input_notfall_dir_0 = raw_npy_dir_notfall
        output_dir_0 = "/home/xubuntu/Desktop/sensor_data/"
        # subprocess.call("python3 thresholding_wrapper.py --program_abs_dir {0} --input_fall_dir {1} --input_not_fall_dir {2} --output_dir {3} --width_of_rim_for_comparison {4} --width_of_ignored_pixels {5} --number_of_parallel_processes {6}".format(program_abs_dir_0, input_fall_dir_0, input_notfall_dir_0, output_dir_0, width_of_rim_for_comparison_0, width_of_ignored_pixels_0, number_of_parallel_processes), shell=True)
        # print("{0}_{1}: Thresholding done.".format(width_of_rim_for_comparison_0, width_of_ignored_pixels_0))

        # run retrieve coordinates
        input_fall_dir_1 = os.path.join(output_dir_0, "{0}_{1}/fall/".format(width_of_rim_for_comparison_0, width_of_ignored_pixels_0))
        input_notfall_dir_1 = os.path.join(output_dir_0, "{0}_{1}/not_fall/".format(width_of_rim_for_comparison_0, width_of_ignored_pixels_0))
        output_dir_1 = "/home/xubuntu/Desktop/sensor_data"
        max_coordinates_1 = 500
        subprocess.call("python3 retrieve_coordinates.py --input_fall_dir {0} --input_notfall_dir {1} --output_dir {2} --max_coordinates {3} --number_of_parallel_processes {4}".format(input_fall_dir_1, input_notfall_dir_1, output_dir_1, max_coordinates_1, number_of_parallel_processes), shell=True)
        print("{0}_{1}: Retrieving coordinates done.".format(width_of_rim_for_comparison_0, width_of_ignored_pixels_0))
    
        # train and retrieve test results
        from svm import main
        input_fall_dir_2 = os.path.join(output_dir_1, "retrieve_coordinates_{0}coordinates/fall/".format(max_coordinates_1))
        input_not_fall_dir_2 = os.path.join(output_dir_1, "retrieve_coordinates_{0}coordinates/not_fall/".format(max_coordinates_1))
        percentage_train_2 = 0.7
        result = main(input_fall_dir_2, input_not_fall_dir_2, percentage_train_2, number_of_parallel_processes)
    
        with open(overall_output, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([width_of_rim_for_comparison_0, width_of_ignored_pixels_0, max_coordinates_1, result[0], result[1], result[2], result[3]])
        print("{0}_{1}: Training and testing SVM done.".format(width_of_rim_for_comparison_0, width_of_ignored_pixels_0))
