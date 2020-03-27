import numpy as np
import argparse



def thresholding_data(output_dir, sample_number):
    data = None
    if sample_number == 1:
        data = np.array([
            [1,2,3,4,3,2,1],
            [2,3,4,5,4,3,2],
            [3,4,5,6,5,4,3],
            [4,5,6,7,6,5,4],
            [3,4,5,6,5,4,3],
            [2,3,4,5,4,3,2],
            [1,2,3,4,3,2,1]
        ])
    else:
        print("ERROR - Sample data not available for sample_number {0}".format(sample_number))
        
    np.save(output_dir, data)

def identify_features(output_dir, sample_number):
    data = None
    if sample_number == 1:
        data = np.array([
            [0,0,0,0,0,0,0,0,0,0],
            [0,9,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,9,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,3,0,0],
            [0,0,0,0,0,0,0,0,0,0],
        ])
    else:
        print("ERROR - Sample data not available for sample_number {0}".format(sample_number))
        
    np.save(output_dir, data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates sample data for running tests')
    parser.add_argument('--output_filename', dest='output_filename', help="output dir of .npy data file")
    parser.add_argument('--mode', dest='mode', help="sets which program to generate test sample data for")
    parser.add_argument('--sample_number', dest='sample_number', help="sets hardcoded sample data, unique to each program")
    args = parser.parse_args()

    if args.output_filename and args.mode and args.sample_number:
        if int(args.mode) == 1:
            thresholding_data(args.output_filename, int(args.sample_number))
        elif int(args.mode) == 2:
            identify_features(args.output_filename, int(args.sample_number))
        else:
            print("ERROR - Mode number {0} not available".format(args.mode))
    else:
        output_dir = "/home/xubuntu/Desktop/capstone/ml/sample1.npy"
        thresholding_data(output_dir, 1)
        # identify_features(output_dir, 1)

    print("create sample data - done.")