"""
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
import subprocess


falldata_dir = ""
nonfalldata_dir = ""
current_dir = os.path.dirname(os.path.realpath(__file__))
number_of_cores_available = 6



# generate coordinates numpy array
def retrieve_coordinates(process_id, list_of_fall_files, input_fall_folder_dir, list_of_notfall_files, input_notfall_folder_dir, output_dir):
    """
    Run per process. Goes through fixed set of numpy arrays and write non-zero coordinates 
    into a separate numpy array (which is saved to temp/). Returns minimum number of coordinates across all numpy arrays.
    """
    import numpy as np

    temp = [[list_of_fall_files, input_fall_folder_dir], [list_of_notfall_files, input_notfall_folder_dir]]
    total_length = len(list_of_fall_files) + len(list_of_notfall_files)
    min_coordiantes_count = float('inf')
    count = 0

    for each_category in temp:
        for each_file in each_category[0]:
            input_dir = os.path.join(each_category[1], each_file)
            print("Process {0}: processing file {1}/{2}; input_dir: {3}".format(process_id, count, total_length, input_dir))

            input_data = np.load(input_dir)
            for i in range(input_data.shape[0]):
                input_frame = input_data[i]
                coordinates = np.where(input_frame>0)
                print("coordinates, ", coordinates)
                break

            count += 1


if __name__ == "__main__":
    # create temp folder to store metadata (return error if folder alr exist)
    temp_folder_dir = os.path.join(current_dir, "temp")
    if os.path.exists(temp_folder_dir):
        print("ERROR - Temp folder already exist. Please delete before running script")
        exit(0)
    os.mkdir(temp_folder_dir)


    # for testing
    list_of_fall_files = ["sample1.npy"]
    input_fall_folder_dir = "/home/xubuntu/Desktop/capstone/ml/thresholded(comparisonrim_ignoredpixels)/test/1_1/fall/"
    list_of_notfall_files = ["sample2.npy"]
    input_notfall_folder_dir = "/home/xubuntu/Desktop/capstone/ml/thresholded(comparisonrim_ignoredpixels)/test/1_1/not_fall/"
    output_dir = os.path.join(temp_folder_dir, "temp1.npy")

    retrieve_coordinates(1, list_of_fall_files, input_fall_folder_dir, list_of_notfall_files, input_notfall_folder_dir, output_dir)




# cleanup
subprocess.run("rm -rf {0}".format(temp_folder_dir), shell=True)