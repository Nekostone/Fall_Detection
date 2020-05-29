import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
# dataset = pd.read_csv('C:\\Users\\user\\Documents\\Capstone\\bill_authentication.csv')
import argparse
import os
import multiprocessing
import random
import subprocess
import copy

# falldata = np.load('',allow_pickle=True)
# nonfalldata = np.load('C:\\Users\\user\\Documents\\Capstone\\DownsampleData\\downsample2\\20200319_manypeople_2_0.npy',allow_pickle=True)

# fallValues = []
# sumOfFallFrameValues = 0
# for oneFallFrame in falldata[0]:
# oneFallFrame = falldata[0][2]
# for valueArray in oneFallFrame:
#     for value in valueArray:
#         print(value)
#         sumOfFallFrameValues += value
# fallValues.append(sumOfFallFrameValues)
# sumOfFallFrameValues = 0
#print("Fall is: ", fallValues)


# nonfallValues = []
# sumOfNonFallFrameValues = 0
# oneNonFallFrame =nonfalldata[0][2]
#for oneNonFallFrame in nonFallFrame:
# for listOfValues in oneNonFallFrame:
    # for singleValue in listOfValues:
        # sumOfNonFallFrameValues += singleValue
        # print(singleValue) 
# nonfallValues.append(sumOfNonFallFrameValues)
# # sumOfNonFallFrameValues = 0
# print("NonFall is: ",nonfallValues)

# print('hello world')
# dataset.head()

#data range
# X = dataset.iloc[:,0:4].values
# y = dataset.iloc[:, 4].values

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# classifier = RandomForestClassifier()
# classifier.fit(X_train,y_train)
# y_pred = classifier.predict(X_test)

# print(y_pred)
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test,y_pred))



def flatten_array_process_wrapper(process_id, list_of_fall_files, input_fall_dir, list_of_notfall_files, input_notfall_dir, output_fall_dir, output_notfall_dir):
    """
    Wrapper that calls flatten_array and labels individual files
    LABELS: 1 - fall, 0 - not_fall
    output format:
    [
        [[1D array representing coordinates]], ['fall']
    ]
    """
    data = {"fall": [list_of_fall_files, input_fall_dir, output_fall_dir], "not_fall": [list_of_notfall_files, input_notfall_dir, output_notfall_dir]}
    total_length = len(list_of_fall_files) + len(list_of_notfall_files)
    count = 0

    # print("list_of_fall_files: {0}\n".format(list_of_fall_files))
    # print("list_of_notfall_files: {0}".format(list_of_notfall_files))

    for each_category in data:
        for each_file in data[each_category][0]:
            input_dir = os.path.join(data[each_category][1], each_file)
            input_data = np.load(input_dir, allow_pickle=True)
            # print("Process {0}: processing file {1}/{2}; input_dir: {3}".format(process_id, count, total_length, input_dir))

            output_data = flatten_array(input_data)
            if each_category == "fall":
                output_data = [[output_data], [1]]  # apply label, they're wrapped with brackets as they'll be directly input into model
            elif each_category == "not_fall":
                output_data = [[output_data], [0]]  # apply label, they're wrapped with brackets as they'll be directly input into model

            output_dir = os.path.join(data[each_category][2], each_file)
            np.save(output_dir, output_data)

            count += 1


def flatten_array(input_data):
    """
    Flattens each array that represents a file
    :input_data: numpy array
    """
    output = []

    input_data = input_data.flatten(order='C')

    for each_frame in input_data:
        for each_value in each_frame:
            output += each_frame[each_value]

    output = np.array(output).flatten(order='C')

    return output


def main(input_fall_dir, input_notfall_dir, percentage_train, number_of_parallel_processes):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # create temp folder to store metadata (return error if folder alr exist)
    temp_folder_dir = os.path.join(current_dir, "temp")
    if os.path.exists(temp_folder_dir):
        subprocess.call("rm -rf {0}".format(temp_folder_dir), shell=True)
        print("ERROR - folder {0} already exist. Deleting old folder...".format(temp_folder_dir))
    os.mkdir(temp_folder_dir)
    
    temp_fall_dir = os.path.join(temp_folder_dir, "fall/")
    if os.path.exists(temp_fall_dir):
        subprocess.call("rm -rf {0}".format(temp_fall_dir), shell=True)
        print("ERROR - folder {0} already exist. Deleting old folder...".format(temp_fall_dir))
    os.mkdir(temp_fall_dir)

    temp_notfall_dir = os.path.join(temp_folder_dir, "not_fall/")
    if os.path.exists(temp_notfall_dir):
        subprocess.call("rm -rf {0}".format(temp_notfall_dir), shell=True)
        print("ERROR - folder {0} already exist. Deleting old folder...".format(temp_notfall_dir))
    os.mkdir(temp_notfall_dir)
    
    # START - multithreaded implementation
    # split files evenly to be processed by each process
    data = []
    for i in [input_fall_dir, input_notfall_dir]:
        list_of_files = os.listdir(i)
        d = len(list_of_files)//number_of_parallel_processes
        sublist_of_files = []

        j = -1
        for j in range(number_of_parallel_processes-1):
            sublist_of_files.append(list_of_files[j*d:(j+1)*d])
        sublist_of_files.append(list_of_files[(j+1)*d:])

        data.append(sublist_of_files)
    # flatten and label each file's array
    output_file_list = []
    process_list = []
    for i in range(number_of_parallel_processes):
        p = multiprocessing.Process(target=flatten_array_process_wrapper, args=(i, data[0][i], input_fall_dir, data[1][i], input_notfall_dir, temp_fall_dir, temp_notfall_dir))
        p.start()
        process_list.append(p)

    for i in range(number_of_parallel_processes):
        process_list[i].join()
    create train and test npy arrays (both include a similar proportion of fall and nonfall files)
    temp_fall_files = copy.deepcopy(os.listdir(temp_fall_dir))
    temp_notfall_files = copy.deepcopy(os.listdir(temp_notfall_dir))

    for i in range(len(temp_fall_files)):
        temp_fall_files[i] = os.path.join(temp_fall_dir, temp_fall_files[i])
    for i in range(len(temp_notfall_files)):
        temp_notfall_files[i] = os.path.join(temp_notfall_dir, temp_notfall_files[i])

    random.shuffle(temp_fall_files)
    random.shuffle(temp_notfall_files)
    train_fall_count = int(len(temp_fall_files)*percentage_train)
    train_notfall_count = int(len(temp_notfall_files)*percentage_train)

    # print("len(temp_notfall_files[train_notfall_count:]): {0}".format(len(temp_notfall_files[train_notfall_count:])))

    train_files_dir = temp_fall_files[:train_fall_count] + temp_notfall_files[:train_notfall_count]
    test_files_dir = temp_fall_files[train_fall_count:] + temp_notfall_files[train_notfall_count:]

    # print("test_files_dir: {0}".format(test_files_dir))

    train_test_data = []
    for each_category in [train_files_dir, test_files_dir]:
        is_first = True
        to_append_to_main = None
        for each_file in each_category:
            input_array = np.load(each_file, allow_pickle=True)
            if is_first:
                to_append_to_main = input_array
                is_first = False
            else:
                to_append_to_main = np.concatenate((to_append_to_main, input_array), axis=1)
        train_test_data.append(to_append_to_main)
    train_data = train_test_data[0]
    test_data = train_test_data[1]

    subprocess.run("rm -rf {0}".format(temp_folder_dir), shell=True)

    # START TRAINING
    # print("Process MASTER: Training start!")
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(train_data[0].tolist(), train_data[1].tolist())
    # # Test effectiveness of SVM model by using train data
    # correct_count = 0
    # wrong_count = 0
    # for i in range(len(train_data[0])):
    #     predict = svm_model.predict([train_data[0][i]])[0]
    #     actual = train_data[1][i]
    #     if predict == actual:
    #         correct_count += 1
    #     else:
    #         wrong_count += 1
    # print("accuracy: {0}".format(correct_count/(correct_count+wrong_count)))

    # Retrieve confusion matrix values from test data
    ### (true fall, true nonfall, false fall, false nonfall)
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(test_data[0])):
        predict = rf_model.predict([test_data[0][i]])[0]
        actual = test_data[1][i]

        """
        True Positive (TP) : Observation is positive, and is predicted to be positive.
        False Negative (FN) : Observation is positive, but is predicted negative.
        True Negative (TN) : Observation is negative, and is predicted to be negative.
        False Positive (FP) : Observation is negative, but is predicted positive.
        """
        if predict == actual:
            if actual == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if actual == 1:
                false_negative += 1
            else:
                false_positive += 1
    # print("@@@ : {0}".format(predict))   
    return (true_positive, true_negative, false_positive, false_negative)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applies thresholding filter on numpy array file')
    parser.add_argument('--input_fall_dir', required=True, type = str, dest='input_fall_dir', help="Directory of input fall data")
    parser.add_argument('--input_notfall_dir', required=True, type = str, dest='input_notfall_dir', help="Directory of input nonfall data")
    parser.add_argument('--percentage_train', required=True, type = float, dest='percentage_train', help="Percentage of all data that'll be used for training (value between 0 - 1)")
    parser.add_argument('--number_of_parallel_processes', required=True, type = int, dest='number_of_parallel_processes', help="Number of parallel processes to use (in doubt just put 1)")
    args = parser.parse_args()

    result = main(args.input_fall_dir, args.input_notfall_dir, args.percentage_train, args.number_of_parallel_processes)
    # print("result: {0}".format(result))

