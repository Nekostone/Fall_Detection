import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
# dataset = pd.read_csv('C:\\Users\\user\\Documents\\Capstone\\bill_authentication.csv')

# falldata = np.load('C:\\Users\\user\\Documents\\Capstone\\DownsampleData\\downsample2\\20200319_sw_4m_sideways3_1.npy',allow_pickle=True)
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

print("hello world")
