import os
from multiprocessing import Process


program_abs_dir = "/home/xubuntu/Desktop/Fall_Detection/write_to_npy.py"
input_fall_dir = "/home/xubuntu/Desktop/sensor_data/raw/fall/"
input_not_fall_dir = "/home/xubuntu/Desktop/sensor_data/raw/not_fall/"
output_fall_dir = "/home/xubuntu/Desktop/sensor_data/processed/fall/"
output_not_fall_dir = "/home/xubuntu/Desktop/sensor_data/processed/not_fall/"
number_of_parallel_processes = 10

def processfiles(process_id, list_of_fall_files, input_fall_folder_dir, output_fall_folder_dir, list_of_notfall_files, input_notfall_folder_dir, output_notfall_folder_dir):
    import subprocess

    temp = [[list_of_fall_files, input_fall_folder_dir, output_fall_folder_dir], [list_of_notfall_files, input_notfall_folder_dir, output_notfall_folder_dir]]
    total_length = len(list_of_fall_files) + len(list_of_notfall_files)
    count = 0

    for each_category in temp:
        for each_file in each_category[0]:
            if each_file[-3:] == "bin":
                input_dir = os.path.join(each_category[1], each_file)
                output_dir = os.path.join(each_category[2], each_file[:-3]+"npy")
                print("Process {0}: processing file {1}/{2}; input_dir: {3}".format(process_id, count, total_length, input_dir))
                subprocess.call(["python3", program_abs_dir, "--input_filename", input_dir, "--output_filename", output_dir])
            count += 1

if __name__ == '__main__':
    # split files evenly to be processed by each process
    data = []
    for i in [input_fall_dir, input_not_fall_dir]:
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
        p = Process(target=processfiles, args=(i, data[0][i], input_fall_dir, output_fall_dir, data[1][i], input_not_fall_dir, output_not_fall_dir,))
        p.start()
        process_list.append(p)
    
    for i in range(number_of_parallel_processes):
        process_list[i].join()

    print("done.")