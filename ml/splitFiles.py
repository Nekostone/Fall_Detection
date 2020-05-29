#Split files into fall and non fall directories

import numpy as np 
import os
import argparse
import shutil

current_dir = os.path.dirname(os.path.realpath(__file__))
temp_folder_dir = os.path.join(current_dir, "temp")
temp_fall_dir = os.path.join(temp_folder_dir, "fall/")
if os.path.exists(temp_fall_dir):
    subprocess.call("rm -rf {0}".format(temp_fall_dir), shell=True)
    print("ERROR - folder {0} already exist. Deleting old folder...".format(temp_fall_dir))
os.makedirs(temp_fall_dir)
# def main(input_file_dir, output_fall_dir, output_nonfall_dir):
#     falls = []
#     nonfalls = []
#     for filename in os.listdir(input_file_dir):
#         data = np.load(os.path.join(input_file_dir, filename),allow_pickle=True)
#         if data[1] == 0:
#             nonfalls.append(filename)
#         else:
#             falls.append(filename)
#     # print(falls)
#     # print('*************************************')
#     # print(nonfalls)
#     for fallfile in falls:
#         shutil.move(os.path.join(input_file_dir, fallfile), os.path.join(output_fall_dir, fallfile))
#     for nonfallfile in nonfalls:
#         shutil.move(os.path.join(input_file_dir, nonfallfile), os.path.join(output_nonfall_dir, nonfallfile))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Splits file to fall and nonfall')
#     parser.add_argument('--input_file_dir', required=True, type = str, dest='input_file_dir', help="Directory of input data")
#     parser.add_argument('--output_fall_dir', required=True, type = str, dest='output_fall_dir', help="Directory of output fall data")
#     parser.add_argument('--output_nonfall_dir', required=True, type = str, dest='output_nonfall_dir', help="Directory of output nonfall data")
#     args = parser.parse_args()

#     result = main(args.input_file_dir, args.output_fall_dir, args.output_nonfall_dir)
#     # print("result: {0}".format(result))

