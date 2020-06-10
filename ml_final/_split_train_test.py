import random
import os
import numpy as np


def split_train_test(folder_dir, train_percentage, fall_percentage=0.5):
    """
    - Datapoints will be separated into train/nottrain with equal numbers of fall and nonfall in both categories
    - Fall/nonfall data points will be removed from train bucket accordingly to ensure percentage is kept
    - By keeping the ratio of fall/nonfall datapoints constant in test bucket, results are comparable regardless of train_percentage and fall_percentage

    """
    # fill up a list of fall and nonfall filename lists
    fall_filenames = []
    nonfall_filenames = []
    for each_file in os.listdir(folder_dir):
        each_file_dir = os.path.join(folder_dir, each_file)
        input_array = np.load(each_file_dir, allow_pickle=True)
        if input_array[1] == 1:
            fall_filenames.append(each_file)
        elif input_array[1] == 0:
            nonfall_filenames.append(each_file)

    # generate number of fall and nonfall data used for training
    train_falldata_total = int(len(fall_filenames)*train_percentage)
    train_nonfalldata_total = int(len(nonfall_filenames)*train_percentage)
    test_falldata_total = int(len(fall_filenames)*(1-train_percentage))
    test_nonfalldata_total = int(len(nonfall_filenames)*(1-train_percentage))
    train_fall_initial_ratio = train_falldata_total/(train_falldata_total+train_nonfalldata_total)

    # ensure train data has correct percentage of fall
    if train_fall_initial_ratio > fall_percentage:
        print("old train_falldata_total: {0}".format(train_falldata_total))
        train_falldata_total = int((fall_percentage*train_nonfalldata_total)/(1-fall_percentage))
        print("new train_falldata_total: {0}".format(train_falldata_total))
    elif train_fall_initial_ratio < fall_percentage:
        print("old train_nonfalldata_total: {0}".format(train_nonfalldata_total))
        train_nonfalldata_total = int((train_falldata_total-fall_percentage*train_falldata_total)/fall_percentage)
        print("new train_nonfalldata_total: {0}".format(train_nonfalldata_total))

    # fill up train and test filename lists
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    _populate_array_list(train_x, train_y, train_falldata_total, fall_filenames, folder_dir)
    _populate_array_list(test_x, test_y, test_falldata_total, fall_filenames, folder_dir)
    _populate_array_list(train_x, train_y, train_nonfalldata_total, nonfall_filenames, folder_dir)
    _populate_array_list(test_x, test_y, test_nonfalldata_total, nonfall_filenames, folder_dir)

    return train_x, train_y, test_x, test_y

def _populate_array_list(list_x, list_y, count, filenames_list, temp_folder_dir):
    for i in range(count):
        each_file = random.choice(filenames_list)
        filenames_list.remove(each_file)
        each_file_dir = os.path.join(temp_folder_dir, each_file)
        input_array = np.load(each_file_dir, allow_pickle=True)
        list_x.append(input_array[0])
        list_y.append(input_array[1])
    
