import numpy as np


x = np.array([
        [
            [1,2,3],
            [1,2,3],
            [1,2,3]
        ],
        [
            [1,2,3],
            [1,2,3],
            [1,2,3]
        ],
        [
            [1,2,3],
            [1,2,3],
            [1,2,3]
        ],
    ])

print("old shape: {0}".format(x.shape))


for i in range(x.shape[0]):
    x[i,:,:] = np.transpose(x[i,:,:])

print("x: {0}".format(x))
print("new shape: {0}".format(x.shape))
