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
        ]
    ])

for i in range(x.shape[0]):
    x[i,:,:] = np.transpose(x[i,:,:])

print(x)
