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
from models.svm import create_train_test
import pickle
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


def randomforest(folder_dir, train_percentage):
    train_x, train_y, test_x, test_y = create_train_test(folder_dir, train_percentage)

    # rfc = RandomForestClassifier()
    # print("Fitting Random Forest")
    # rfc.fit(train_x, train_y)
    # print("Complete fitting")
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    print("Testing pickle")
    rfc = pickle.load(open("/home/chongyicheng/Capstone/Fall_Detection/randomforest.sav", 'rb'))
    isFirst = True
    for i in range(len(test_x)):
        predict = rfc.predict([test_x[i]])
        actual = test_y[i]
        if isFirst == True:
            print("Array for frames look like this")
            print(test_x[i])
            isFirst = False

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

    # filename = 'randomforest.sav'
    # pickle.dump(rfc, open(os.path.join("/home/chongyicheng/Capstone/Fall_Detection",filename), 'wb'))

    return (true_positive, true_negative, false_positive, false_negative)

    
