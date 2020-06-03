from sklearn import svm as svm_model
import time
import numpy as np
import pickle

def svm(train_x, train_y, test_x, test_y, weights_dir):
    model = svm_model.SVC()
    model.fit(train_x, train_y)

    # save model in bytes
    weight_bytes = pickle.dumps(model)
    with open(weights_dir, "wb+") as writefile:
        writefile.write(weight_bytes)

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
    


