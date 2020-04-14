"""
Actually the SVM

SVM is classically 2D, so we'll flatten the entire array of coordinates that represents a file
"""

import numpy as np


def flatten_array(input_data):
    """
    Flattens each array that represents a file
    :input_data: numpy array
    """
    output = []

    for each_file in input_data:
        to_insert = []
        for i in range(1,len(each_file)):
            to_insert.append(each_file[i])
        to_insert = np.array(to_insert)
        to_insert = to_insert.flatten("C")
        output.append(to_insert)
    
    output = np.array(output)

    return output

def generate_train_test(fall_data, notfall_data):
    train = None
    test = None


    return train, test

def svm_train():
    pass

def svm_test():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applies thresholding filter on numpy array file')
    parser.add_argument('--input_fall_dir', required=True, type = str, dest='input_fall_dir', help="Directory of input fall data")
    parser.add_argument('--input_notfall_dir', required=True, type = str, dest='input_notfall_dir', help="Directory of input nonfall data")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # create temp folder to store metadata (return error if folder alr exist)
    temp_folder_dir = os.path.join(current_dir, "temp")
    if os.path.exists(temp_folder_dir):
        print("ERROR - Temp folder already exist. Please delete before running script")
        exit(0)
    os.mkdir(temp_folder_dir)

    raw_fall_data = np.load(args.input_fall_dir)
    raw_notfall_data = np.load(args.input_notfall_dir)

    flat_fall_data = flatten_array(raw_fall_data)
    flat_notfall_data = flatten_array(raw_notfall_data)

    train, test = generate_train_test(flat_fall_data, flat_notfall_data)
    
    
