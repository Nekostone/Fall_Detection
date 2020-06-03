import numpy as np
import argparse
import csv
import sys
import os



def process_fall_data(input_array: np.array, starting_frame: int, ending_frame: int) -> np.array:
    output_array = input_array[starting_frame:ending_frame,:,:]
    output_array = np.array([output_array, 1])
    return output_array

def process_nonfall_data(input_array: np.array) -> (np.array, np.array, np.array):
    first = np.array([input_array[:50], 0])
    second = np.array([input_array[50:100], 0])
    third = np.array([input_array[100:150], 0])
    return (first, second, third)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Labels and shapes data for filtering and training/testing')
    parser.add_argument('--input_folder', required=True, help='Folder containing the raw .npy files of each take')
    parser.add_argument('--input_csv', help='DIR of CSV file containing labellings of when fall happens')
    parser.add_argument('--output_folder', required=True, help='Output folder to dump labelled data in (containing same names as original input files)')
    parser.add_argument('--is_nonfall_data', action='store_true', help='If set, all files in input folder will be treated as pure nonfalldata')
    args = parser.parse_args()

    if args.is_nonfall_data:
        if args.input_csv != None:
            exit('Flag \'is_nonfall_data\' is on and \'input_csv\' has been specified. DO not specify \'input_csv\' flag if you\'re loading nonflal data')

        for each_file in os.listdir(args.input_folder):
            if each_file[-4:] != '.npy':
                continue

            each_file_dir = os.path.join(args.input_folder, each_file)
            input_array = np.load(each_file_dir)
            output_nparrays = process_nonfall_data(input_array)

            for i in range(3):
                output_file_dir = os.path.join(args.output_folder, each_file[:-4]+'_{0}.npy'.format(i))
                np.save(output_file_dir, output_nparrays[i])
    else:
        if args.input_csv == None:
            exit('Flag \'input_csv\' requires a dir to the CSV file containing labelling information for fall data')

        # runs checks on CSV file
        # returns error if file names don't end with .npy
        with open(args.input_csv) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in spamreader:
                if row[0][-4:] != '.npy':
                    exit('Ensure that file {0} in label file {1} ends with \'.npy\''.format(row[0], args.input_csv))

                try:
                    int(row[1])
                    int(row[2])
                except ValueError:
                    exit('Ensure that rows with non int starting and ending frames should be deleted')

                if int(row[2]) - int(row[1]) != 50:
                    exit('Ensure that ending frame is 50 frames ahead of starting frame')

        # assuming from here all checks passed
        with open(args.input_csv) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in spamreader:
                each_file = row[0]
                starting_frame = int(row[1])
                ending_frame = int(row[2])

                each_file_dir = os.path.join(args.input_folder, each_file)
                input_array = np.load(each_file_dir)
                output_array = process_fall_data(input_array, starting_frame, ending_frame)

                output_file_dir = os.path.join(args.output_folder, each_file)
                np.save(output_file_dir, output_array)






























