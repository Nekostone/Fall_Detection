import numpy as np

x = np.array([
    [1,2,3,4,3,2,1],
    [2,3,4,5,4,3,2],
    [3,4,5,6,5,4,3],
    [4,5,6,7,6,5,4],
    [3,4,5,6,5,4,3],
    [2,3,4,5,4,3,2],
    [1,2,3,4,3,2,1]])

current_dir = "/home/xubuntu/Desktop/capstone/ml/"

np.save(current_dir+"sample1.npy", x)

print("create sample data - done.")
