#! usr/bin/env python

import numpy as np
import pickle
from preprocess_actualdata import preprocess
from _remove_center import remove_center

test_file = r"D:\Documents\SUTD\Capstone\Tests\Live\Sat_04_Jul_2020_08_43_08.npy"
recorded_file = r"D:\Documents\SUTD\Capstone\Tests\Live\Sat_04_Jul_2020_08_43_08_vals.npy"
svm_weights = r"D:\Documents\SUTD\Capstone\Ml_Data\range_rangeDelta_doppDelta_94.pickle"

def cfar(arr):
    # one side
    train = 5
    guard = 3
    p = 1.4

    train_size = 2*(train + guard) + 1
    guard_size = 2* guard +1

    output_arr = np.zeros((128,64))
    arr =np.pad(arr,train_size+guard_size,mode="mean")

    for row in range(128):
        for col in range(64):
            test_cells = arr[row:row+train_size, col:col+train_size] 
            guard_cells = arr[row+train: row + train_size-train, col+train:col+train_size-train]

            ave_noise = (np.sum(test_cells) - np.sum(guard_cells))/(train_size**2 - guard_size**2)
            cut = arr[row+train_size+guard_size, col+train_size+guard_size]

            if cut > ave_noise * p:
                output_arr[row,col] = cut

            # else:
            #      output_arr[row,col] = cut//3 

    return output_arr

with open(svm_weights, "rb") as readfile:
    model = pickle.loads(readfile.read())

# Load test file and recorded file 
test_data = np.load(test_file, allow_pickle=True)/512
actual = np.load(recorded_file, allow_pickle=True)

print("Computing CFAR...")
test_data = [cfar(i) for i in test_data]
for i in test_data:
    i[:,31:34] = 0

print("Computing Energy...")
energy = [np.sum(j) for j in test_data]

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

print("Test start")
# pipe data into model for testing
for i in range(len(test_data)-size):
    print(i)
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

print(len(results))

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

acc = (true_neg + true_pos)/(true_pos + true_neg + false_neg + false_pos)
acc2 = (true_neg2 + true_pos2)/(true_pos2 + true_neg2 + false_neg2 + false_pos2)
print("Before: True positive:{0}, True negative:{1}, False positive:{2}, False negative:{3}, Accuracy:{4}".format(true_pos, true_neg, false_pos, false_neg, acc))
print("True positives: {}".format(frames_true_pos))
print("After: True positive:{0}, True negative:{1}, False positive:{2}, False negative:{3}, Accuracy:{4}".format(true_pos2, true_neg2, false_pos2, false_neg2, acc2))
print("After: False positives: {}".format(frames_false_pos2))
