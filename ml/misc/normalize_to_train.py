import numpy as np

def normalize_to_train(train_x, train_y, test_x, test_y):
    mean = np.mean(train_x)
    std = np.std(train_x)

    # print("mean: {0}; std: {1}".format(mean, std))

    for i in range(len(train_x)):
        train_x[i] = (train_x[i] - mean) / std

    for i in range(len(test_x)):
        test_x[i] = (test_x[i] - mean) / std

    return train_x, train_y, test_x, test_y
