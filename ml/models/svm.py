from sklearn import svm as svm_model
import random
import os
import numpy as np

def svm(folder_dir, train_percentage):
    train_x, train_y, test_x, test_y = create_train_test(folder_dir, train_percentage)

    model = svm_model.SVC()
    model.fit(train_x, train_y)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(test_x)):
        predict = model.predict([test_x[i]])
        actual = test_y[i]

        """
        True Positive (TP) : Observation is positive, and is predicted to be positive.
        False Positive (FP) : Observation is negative, but is predicted positive.
        True Negative (TN) : Observation is negative, and is predicted to be negative.
        False Negative (FN) : Observation is positive, but is predicted negative.
        """

        if predict == actual:
            if predict == 1:
                true_positive += 1
            elif predict == 0:
                true_negative += 1
            else:
                print("??? - {0}".format(predict))
        else:
            if predict == 1:
                false_positive += 1
            elif predict == 0:
                false_negative += 1
            else:
                print("??? - {0}".format(predict))

    return (true_positive, true_negative, false_positive, false_negative)
    


def create_train_test(folder_dir, train_percentage):
    """
    An equal percentage of fall and nonfall data points will fall under training dataset and testing dataset respectively
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

    # fill up train and test filename lists
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(train_falldata_total):
        each_file = random.choice(fall_filenames)
        fall_filenames.remove(each_file)
        each_file_dir = os.path.join(folder_dir, each_file)

        input_array = np.load(each_file_dir, allow_pickle=True)
        train_x.append(input_array[0])
        train_y.append(input_array[1])

    test_fall_count = 0
    for each_testfile in fall_filenames:
        each_file_dir = os.path.join(folder_dir, each_testfile)
        input_array = np.load(each_file_dir, allow_pickle=True)
        test_x.append(input_array[0])
        test_y.append(input_array[1])

        test_fall_count += 1

    for i in range(train_nonfalldata_total):
        each_file = random.choice(nonfall_filenames)
        nonfall_filenames.remove(each_file)
        each_file_dir = os.path.join(folder_dir, each_file)

        input_array = np.load(each_file_dir, allow_pickle=True)
        train_x.append(input_array[0])
        train_y.append(input_array[1])

    test_nonfall_count = 0
    for each_testfile in nonfall_filenames:
        each_file_dir = os.path.join(folder_dir, each_testfile)
        input_array = np.load(each_file_dir, allow_pickle=True)
        test_x.append(input_array[0])
        test_y.append(input_array[1])

        test_nonfall_count += 1

    print("test_fall_count: {0}, test_nonfall_count: {1}".format(test_fall_count, test_nonfall_count))
    print("len(train_x): {0}; len(test_x): {1}".format(len(train_x), len(test_x)))
    

    return train_x, train_y, test_x, test_y
