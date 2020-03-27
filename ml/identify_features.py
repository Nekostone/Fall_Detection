import numpy as np
import argparse


def main(input_dir, mode, val):
    """
    mode number - return coordinates of points that:
    1 - are equivalent or higher than an absolute value
    2 - are of the equivalent or higher than the provided standard deviation value

    returns an array of (x, y) coordinates
    """
    input_data = np.load(input_dir)
    output = None

    # print("input_data: \n", input_data)

    if mode == 1:
        raw_output = np.where(input_data >= val)
        reshaped_output = []
        for each_array in raw_output:
            temp = np.reshape(each_array, (each_array.shape[0], 1))
            reshaped_output.append(temp)

        output = np.concatenate((reshaped_output[1], reshaped_output[0]), axis=1)
    elif mode == 2:
        mean = np.mean(input_data)
        std_dev = np.var(input_data)
        raw_output = np.where((input_data-mean)/std_dev >= val)

        reshaped_output = []
        for each_array in raw_output:
            temp = np.reshape(each_array, (each_array.shape[0], 1))
            reshaped_output.append(temp)

        # print("mean: {0}; std dev: {1}".format(mean, std_dev))

        output = np.concatenate((reshaped_output[1], reshaped_output[0]), axis=1)
    else:
        print("ERROR - Mode number {0} not available".format(mode))
    
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates sample data for running tests')
    parser.add_argument('--input_filename', dest='input_filename', help="input dir of .npy data file")
    args = parser.parse_args()

    if args.input_filename:
        main(args.input_filename)
    else:
        output_dir = "/home/xubuntu/Desktop/capstone/ml/sample1.npy"
        # print(main(output_dir, 1, 2))
        print(main(output_dir, 2, 1))
