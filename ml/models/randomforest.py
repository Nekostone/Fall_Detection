import pandas as pd 
import numpy as np
import time  
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

import pickle

def randomforest(train_x, train_y, test_x, test_y):

    rfc = RandomForestClassifier()
    print("Fitting Random Forest")
    rfc.fit(train_x, train_y)
    print("Complete fitting")
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    #print("Testing pickle")
    #rfc = pickle.load(open("/home/chongyicheng/Capstone/Fall_Detection/randomforest_iter2.sav", 'rb'))
    isFirst = True

    start_time = time.time()
    for i in range(len(test_x)):
        predict = rfc.predict([test_x[i]])

        actual = test_y[i]
        if isFirst == True:
            # print("length for frames")
            # print(len(test_x[i]))
            isFirst = False
        print("{}: {}".format(i,time.time()-start_time))
        """
        True Positive (TP) : Observation is positive, and is predicted to be positive.
        False Positive (FP) : Observation is negative, but is predicted positive.
        True Negative (TN) : Observation is negative, and is predicted to be negative.
        False Negative (FN) : Observation is positive, but is predicted negative.
        """

        # if predict == actual:
        #     if predict == 1:
        #         true_positive += 1
        #     elif predict == 0:
        #         true_negative += 1
        #     else:
        #         print("??? - {0}".format(predict))
        # else:
        #     if predict == 1:
        #         false_positive += 1
        #     elif predict == 0:
        #         false_negative += 1
        #     else:
        #         print("??? - {0}".format(predict))
    # print("{}: {}".format(i,time.time()-start_time))
    # filename = 'randomforest_iter2.sav'
    # pickle.dump(rfc, open(os.path.join("/home/chongyicheng/Capstone/Fall_Detection",filename), 'wb'))

    return (true_positive, true_negative, false_positive, false_negative)

    
