import numpy as np
import argparse



def thresholding_data(output_dir, sample_number):
    data = None
    if sample_number == 1:
        data = np.array([[
            [1,2,3,4,3,2,1],
            [2,3,4,5,4,3,2],
            [3,4,5,6,5,4,3],
            [4,5,6,7,6,5,4],
            [3,4,5,6,5,4,3],
            [2,3,4,5,4,3,2],
            [1,2,3,4,3,2,1]
        ]])
    elif sample_number == 2:
        # used to test thresholding_wrapper
        data = np.array([
            [
                [1,2,3,4,3,2,1],
                [2,3,4,5,4,3,2],
                [3,4,5,6,5,4,3],
                [4,5,6,7,6,5,4],
                [3,4,5,6,5,4,3],
                [2,3,4,5,4,3,2],
                [1,2,3,4,3,2,1]
            ],
            [
                [1,2,3,4,3,2,1],
                [2,3,4,5,4,3,2],
                [3,4,5,6,5,4,3],
                [4,5,6,7,6,5,4],
                [3,4,5,6,5,4,3],
                [2,3,4,5,4,3,2],
                [1,2,3,4,3,2,1]
            ]
        ])
    else:
        print("ERROR - Sample data not available for sample_number {0}".format(sample_number))
        exit(0)
        
    np.save(output_dir, data)

def identify_features(output_dir, sample_number):
    data = None
    if sample_number == 1:
        data = np.array([[
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
        ]])
    else:
        print("ERROR - Sample data not available for sample_number {0}".format(sample_number))
        exit(0)
        
    np.save(output_dir, data)

def svm(output_dir, sample_number):
    data = None
    if sample_number == 1:
        # for testing the coordinates retrieval part of svm
        data = np.array([
            [
                [1,0,0],
                [1,0,0],
                [1,0,0]
            ],
            [
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ],
        ])
    else:
        print("ERROR - Sample data not available for sample_number {0}".format(sample_number))
        exit(0)

    np.save(output_dir, data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates sample data for running tests')
    parser.add_argument('--output_filename', dest='output_filename', help="output dir of .npy data file")
    parser.add_argument('--mode', dest='mode', help="sets which program to generate test sample data for")
    parser.add_argument('--sample_number', dest='sample_number', help="sets hardcoded sample data, unique to each program")
    args = parser.parse_args()

    if args.mode:
        if int(args.mode) == 1:
            print("create_sample_data - Generating identify features test data...")
            if args.output_filename and args.sample_number:
                identify_features(args.output_filename, int(args.sample_number))
            else:
                output_dir = "/home/xubuntu/Desktop/capstone/ml/sample1.npy"
                identify_features(output_dir, 1)

        elif int(args.mode) == 2:
            print("create_sample_data - Generating thresholding test data...")
            if args.output_filename and args.sample_number:
                thresholding_data(args.output_filename, int(args.sample_number))
            else:
                output_dir = "/home/xubuntu/Desktop/capstone/ml/test/fall/sample1.npy"
                thresholding_data(output_dir, 2)
                output_dir = "/home/xubuntu/Desktop/capstone/ml/test/not_fall/sample2.npy"
                thresholding_data(output_dir, 2)

        elif int(args.mode) == 3:
            print("create_sample_data - Generating svm test data...")
            if args.output_filename and args.sample_number:
                svm(args.output_filename, int(args.sample_number))
            else:
                output_dir = "/home/xubuntu/Desktop/capstone/ml/thresholded(comparisonrim_ignoredpixels)/test/1_1/fall/sample1.npy"
                svm(output_dir, 1)
                output_dir = "/home/xubuntu/Desktop/capstone/ml/thresholded(comparisonrim_ignoredpixels)/test/1_1/not_fall/sample2.npy"
                svm(output_dir, 1)
        else:
            print("ERROR - Mode number {0} not available".format(args.mode))
    else:
        output_dir = "/home/xubuntu/Desktop/capstone/ml/sample1.npy"
        thresholding_data(output_dir, 1)
