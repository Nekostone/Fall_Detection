import numpy as np
import argparse

def main(input_dir, output_dir, width_of_ignored_pixels, width_of_rim_for_comparison):
    input_data = np.load(input_dir)
    # print("input_data (before threshold): \n", input_data)
    output_data = np.zeros((input_data.shape[0], input_data.shape[1]))

    # applying padding
    input_data = np.pad(input_data, width_of_ignored_pixels+width_of_rim_for_comparison, 'constant', constant_values=0)
    # print("input_data (after padding): \n", input_data)

    length_of_kernel = width_of_ignored_pixels*2 + width_of_rim_for_comparison*2 + 1
    kernel = np.zeros((length_of_kernel-(2*width_of_rim_for_comparison),length_of_kernel-(2*width_of_rim_for_comparison)))
    kernel = np.pad(kernel, width_of_rim_for_comparison, 'constant', constant_values=1)

    input_data_org_breadth = input_data.shape[0]-(2*width_of_ignored_pixels)-(2*width_of_rim_for_comparison)
    input_data_org_length = input_data.shape[1]-(2*width_of_ignored_pixels)-(2*width_of_rim_for_comparison)

    for y in range(input_data_org_breadth):
        for x in range(input_data_org_length):
            # generate mask
            mask = np.copy(kernel)
            mask = np.pad(mask, ((y, input_data.shape[0]-length_of_kernel-y), (x, input_data.shape[1]-length_of_kernel-x)), 'constant', constant_values=0)

            # apply mask to input_data and sum output
            mask_output = np.multiply(mask, input_data)
            # print("mask_output: \n", mask_output)

            num_of_elements_retrieved = 0
            for i in range(width_of_rim_for_comparison):
                num_of_elements_retrieved += (length_of_kernel-1-(2*i))*4

            # if sum more than center of kernel, new value is zero, otherwise save the value
            mask_output_avg = np.sum(mask_output)/num_of_elements_retrieved
            # print("y: {0}; x: {1}; mask_output_avg: {2}; input_data[y+width_of_ignored_pixels+width_of_rim_for_comparison][x+width_of_ignored_pixels+width_of_rim_for_comparison]: {3}".format(y, x, mask_output_avg, input_data[y+width_of_ignored_pixels+width_of_rim_for_comparison][x+width_of_ignored_pixels+width_of_rim_for_comparison]))
            if input_data[y + width_of_ignored_pixels + width_of_rim_for_comparison][x + width_of_ignored_pixels + width_of_rim_for_comparison] >= mask_output_avg:
                output_data[y][x] = input_data[y + width_of_ignored_pixels + width_of_rim_for_comparison][x + width_of_ignored_pixels + width_of_rim_for_comparison]

    print("input_data: \n{0}".format(input_data))
    print("output_data: \n{0}".format(output_data))

    np.save(output_dir, output_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Applies thresholding filter on numpy array file')
    parser.add_argument('--input_filename', dest='input_filename', help="input dir of .npy data file")
    parser.add_argument('--output_filename', dest='output_filename', help="output dir of .npy data file")
    parser.add_argument('--width_of_ignored_pixels', dest='width_of_ignored_pixels', help="width of pixels ignored in threshold")
    parser.add_argument('--width_of_rim_for_comparison', dest='width_of_rim_for_comparison', help="width of pixels to compare with the center pixel of mask")
    args = parser.parse_args()

    if args.input_filename and args.output_filename and args.width_of_ignored_pixels and args.width_of_rim_for_comparison:
        main(args.input_filename, args.output_filename, args.width_of_ignored_pixels, args.width_of_rim_for_comparison)
    else:
        main("/home/xubuntu/Desktop/capstone/ml/sample1.npy", "/home/xubuntu/Desktop/capstone/ml/sample1_out.npy", 1, 2)