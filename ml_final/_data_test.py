#! usr/bin/env python

import numpy as np
import pickle
from preprocess_actualdata import preprocess
from _remove_center import remove_center
import time
from scipy.signal import convolve

test_file = r"D:\Documents\SUTD\Capstone\Tests\Live\Sat_04_Jul_2020_08_43_08.npy"
recorded_file = r"D:\Documents\SUTD\Capstone\Tests\Live\Sat_04_Jul_2020_08_43_08_vals.npy"
svm_weights = r"D:\Documents\SUTD\Capstone\Fall_Detection\range_rangeDelta_doppDelta_94.pickle"
pca_weights = r"D:\Documents\SUTD\Capstone\Ml_Data\pca_weights.pickle"


def fft_cfar(arr):
    train = 5
    guard = 3
    p = 1.4

    train_size = 2*(train + guard) + 1
    guard_size = 2* guard +1
    arr2 = np.pad(arr,train + guard,mode="mean")

    # build kernel
    kernel = np.zeros((7,7))
    kernel = np.pad(kernel, 5, constant_values = 1)

    # perform cfar operation
    ave_noise = convolve(arr2, kernel, mode="same")[train+guard : train+guard+128, train+guard: train+guard+64] /(train_size**2 - guard_size**2)
    
    truth = np.greater_equal(arr, p* ave_noise)
    output = np.where(truth, arr, 0)

    return output

def cfar(arr, start_row_idx = 0, end_row_idx = 128):
    # one side
    train = 5
    guard = 3
    p = 1.4

    train_size = 2*(train + guard) + 1
    guard_size = 2* guard +1

    output_arr = np.zeros((end_row_idx-start_row_idx,64))
    arr =np.pad(arr, train + guard ,mode="mean")

    for row in range(start_row_idx, end_row_idx):
        for col in range(64):
            test_cells = arr[row:row+train_size, col:col+train_size] 
            guard_cells = arr[row + train: row + train_size-train, col+train:col+train_size-train]

            big_square = np.sum(test_cells)
            small_square = np.sum(guard_cells)
            ave_noise = (big_square - small_square)/(train_size**2 - guard_size**2)
            cut = arr[row + train + guard , col + train +guard]

            if cut > ave_noise * p:
               output_arr[row-start_row_idx,col] = cut

    return output_arr



with open(svm_weights, "rb") as readfile:
    model = pickle.loads(readfile.read())

with open(pca_weights, "rb") as readfile2:
    pca = pickle.loads(readfile2.read())

# Load test file and recorded file 
test_data = np.load(test_file, allow_pickle=True)/512
actual = np.load(recorded_file, allow_pickle=True)

results = []
results2 = []

# counters
false_pos = 0
true_pos= 0
false_neg = 0
true_neg = 0

false_pos2 = 0
true_pos2 = 0
false_neg2 = 0
true_neg2 = 0

frames_false_pos = []
frames_false_neg = []
frames_true_pos = []

frames_false_pos2 = []

threshold = 100
size = 20

start = time.time()

print("Computing CFAR...")
ping1 = time.time()
test_data = [cfar(i) for i in test_data]
for i in test_data:
    i[:,31:34] = 0
pong1 = time.time()

print("Computing Energy...")
energy = [np.sum(j) for j in test_data]

print("Test start")
# pipe data into model for testing

ping = time.time()
for i in range(len(test_data)-size):
    current_test_data = test_data[i:i+5]
    preprocessed = preprocess(current_test_data)
    output = model.predict(preprocessed)
    results.append(output)

    if output == 1:
        look_ahead = energy[i+5:i+size]
        if np.sum(look_ahead)/(size-5) >= threshold:
            results2.append(0) 

        else:
            results2.append(output)
    else:
        results2.append(output)

# compile results
for idx in range(len(results)):
    if actual[idx] == 1 and results[idx] == 1:
        true_pos += 1
        frames_true_pos.append(idx)
    
    elif actual[idx] == 0 and results[idx] == 1:
        false_pos += 1
        frames_false_pos.append(idx)

    elif actual[idx] == 0 and results[idx] == 0:
        true_neg += 1

    elif actual[idx] == 1 and results[idx] == 0:
        false_neg += 1
        frames_false_neg.append(idx)

for idx in range(len(results2)):
    if actual[idx] == 1 and results2[idx] == 1:
        true_pos2 += 1
    
    elif actual[idx] == 0 and results2[idx] == 1:
        false_pos2 += 1
        frames_false_pos2.append(idx)

    elif actual[idx] == 0 and results2[idx] == 0:
        true_neg2 += 1

    elif actual[idx] == 1 and results2[idx] == 0:
        false_neg2 += 1

end = time.time()
ave = (end-start)/len(results2)

acc = (true_neg + true_pos)/(true_pos + true_neg + false_neg + false_pos)
acc2 = (true_neg2 + true_pos2)/(true_pos2 + true_neg2 + false_neg2 + false_pos2) 
print("Before: True positive:{0}, True negative:{1}, False positive:{2}, False negative:{3}, Accuracy:{4}".format(true_pos, true_neg, false_pos, false_neg, acc))
print("True positives: {}".format(frames_true_pos))
print("After: True positive:{0}, True negative:{1}, False positive:{2}, False negative:{3}, Accuracy:{4}".format(true_pos2, true_neg2, false_pos2, false_neg2, acc2))
print("After: False positives: {}".format(frames_false_pos2))
print("Average time per frame: {}".format(ave))

print("Total cfar time taken: {}".format(pong1-ping1))
print("Total svm time taken: {}".format(end-ping))