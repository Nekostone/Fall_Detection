import numpy as np

x = np.array([
    [
        [1,1,1],
        [2,1,1],
        [3,1,1]
    ],
    [
        [4,1,1],
        [5,1,1],
        [6,1,1]
    ],
    [
        [7,1,1],
        [8,1,1],
        [9,1,1]
    ],
])



print(x[:,:2,:])
print(np.concatenate((x[:,:1,:], x[:,2:,:]), axis=1))
