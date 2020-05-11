"""
Misc script used for dev
"""


import numpy as np

def main():
    first_input = np.load('/home/xubuntu/Desktop/sensor_data/retrieve_coordinates/fall/202003020_jd_4m_chair_radial4.npy', allow_pickle=True)

    print("first_input.shape: {0}".format(first_input.shape))
    print("first_input[0].shape: {0}".format(first_input[0].shape))


if __name__ == '__main__':
    main()