import os
import sys
import json
import subprocess
import numpy as np
import unittest


current_dir = os.path.dirname(os.path.realpath(__file__))

# class RetriveCoordinates_function_positive(unittest.TestCase):
#     temp_folder_dir = os.path.join(current_dir, "temp")
#     samplefalldata_dir = os.path.join(temp_folder_dir, "fall")
#     samplenotfalldata_dir = os.path.join(temp_folder_dir, "not_fall")
#     output_dir = os.path.join(current_dir, "../temp")


#     def setUp(self):
#         # create temp folder to store function metadata
#         if os.path.exists(self.output_dir):
#             print("ERROR - Temp metadata folder already exist. Please delete before running script")
#             exit(0)
#         os.mkdir(self.output_dir)

#         # create temp folder to store output data
#         if os.path.exists(self.temp_folder_dir):
#             print("ERROR - Temp folder to store output data already exist. Please delete before running script")
#             exit(0)
#         os.mkdir(self.temp_folder_dir)

#         # create temp falldata folder
#         if os.path.exists(self.samplefalldata_dir):
#             print("ERROR - Temp falldata folder already exist. Please delete before running script")
#             exit(0)
#         os.mkdir(self.samplefalldata_dir)

#         # create temp nonfalldata folder
#         if os.path.exists(self.samplenotfalldata_dir):
#             print("ERROR - Temp nonfalldata folder already exist. Please delete before running script")
#             exit(0)
#         os.mkdir(self.samplenotfalldata_dir)

#         createdata_file_dir = os.path.join(current_dir, "../../create_sample_data.py")

#         # generate sample fall data
#         output_fall_filename = os.path.join(self.samplefalldata_dir, "sample1.npy")
#         subprocess.run("python3 {0} --output_filename {1} --mode {2} --sample_number {3}".format(createdata_file_dir, output_fall_filename, 3, 1), shell=True)

#         # generate sample nonfall data
#         output_notfall_filename = os.path.join(self.samplenotfalldata_dir, "sample2.npy")
#         subprocess.run("python3 {0} --output_filename {1} --mode {2} --sample_number {3}".format(createdata_file_dir, output_notfall_filename, 3, 1), shell=True)
    
#     def test_main(self):
#         parent_dir = os.path.join(current_dir, "../")
#         sys.path.insert(0, parent_dir)
#         import retrieve_coordinates

#         process_id = 0
#         global_dict = {}
#         list_of_fall_files = ["sample1.npy"]
#         input_fall_folder_dir = self.samplefalldata_dir
#         list_of_notfall_files = ["sample2.npy"]
#         input_notfall_folder_dir = self.samplenotfalldata_dir
#         output_fall_dir = os.path.join(self.output_dir, "retrieved_coordinates_fall")
#         output_notfall_dir = os.path.join(self.output_dir, "retrieved_coordinates_notfall")
#         retrieve_coordinates.retrieve_coordinates(process_id, global_dict, list_of_fall_files, input_fall_folder_dir, list_of_notfall_files, input_notfall_folder_dir, output_fall_dir, output_notfall_dir)

#         # with open(output_dir, "r") as readfile:
#         #     result = json.load(readfile)

#         # to_compare = {
#         #     'fall': {
#         #         'sample1.npy': [
#         #             {
#         #                 'values_list': [1.0, 1.0, 1.0], 
#         #                 'coordinates_dict': {
#         #                     '1.0': [[0, 0], [0, 1], [0, 2]]
#         #                 }
#         #             }, 
#         #             {
#         #                 'values_list': [1.0], 
#         #                 'coordinates_dict': {
#         #                     '1.0': [[1, 1]]}}]
#         #     }, 
#         #     'not_fall': {
#         #         'sample2.npy': [
#         #             {
#         #                 'values_list': [1.0, 1.0, 1.0], 
#         #                 'coordinates_dict': {
#         #                     '1.0': [[0, 0], [0, 1], [0, 2]]
#         #                 }
#         #             }, 
#         #             {
#         #                 'values_list': [1.0], 
#         #                 'coordinates_dict': {'1.0': [[1, 1]]}
#         #             }
#         #         ]
#         #     }
#         # }

#         # self.assertEqual(result, to_compare)

#         self.assertTrue(True)
    
#     def tearDown(self):
#         # delete sample data temp folder and contents
#         if os.path.exists(self.temp_folder_dir):
#             subprocess.run("rm -rf {0}".format(self.temp_folder_dir), shell=True)
#         else:
#             print("ERROR - sample data temp folder does not exist.")
#             exit(0)

#         # delete output temp folder and contents
#         if os.path.exists(self.output_dir):
#             subprocess.run("rm -rf {0}".format(self.output_dir), shell=True)
#         else:
#             print("ERROR - output temp folder does not exist.")
#             exit(0)


class MinimumCoordinates_function_positive(unittest.TestCase):
    sampledata_dir = os.path.join(current_dir, "temp")
    samplefalldata_dir = os.path.join(sampledata_dir, "fall/")
    samplenotfalldata_dir = os.path.join(sampledata_dir, "not_fall/")
    sample_file_name = "sample1.npy"
    sample_file_dir = os.path.join(samplefalldata_dir, sample_file_name)


    def setUp(self):
        # create temp folder to store input data
        if os.path.exists(self.sampledata_dir):
            print("ERROR - Temp input data folder already exist. Please delete before running script")
            exit(0)
        os.mkdir(self.sampledata_dir)

        # create temp folder to store input data
        if os.path.exists(self.samplefalldata_dir):
            print("ERROR - Temp input data folder already exist. Please delete before running script")
            exit(0)
        os.mkdir(self.samplefalldata_dir)

        # create temp folder to store input data
        if os.path.exists(self.samplenotfalldata_dir):
            print("ERROR - Temp input data folder already exist. Please delete before running script")
            exit(0)
        os.mkdir(self.samplenotfalldata_dir)

        # save sample data
        sampledata = [
            {
                'values_list': [1,1,3,2],
                'coordinates_dict': {
                    '1': [
                        [1,1],
                        [1,2]
                    ],
                    '2': [
                        [1,3]
                    ],
                    '3': [
                        [1,4]
                    ]
                }
            },
            {
                'values_list': [1],
                'coordinates_dict': {
                    '1': [
                        [1,1]
                    ]
                }
            }
        ]
        np.save(self.sample_file_dir, np.array(sampledata))
    
    def testMain(self):
        parent_dir = os.path.join(current_dir, "../")
        sys.path.insert(0, parent_dir)
        import retrieve_coordinates

        retrieve_coordinates.minimum_coordinates(0, 1, [self.sample_file_name], self.samplefalldata_dir, [], self.samplenotfalldata_dir)

        result = np.load(self.sample_file_dir, allow_pickle=True)
        print("result: {0}".format(result))

        self.assertTrue(True)
    
    def tearDown(self):
        # delete temp folder and contents
        if os.path.exists(self.sampledata_dir):
            subprocess.run("rm -rf {0}".format(self.sampledata_dir), shell=True)
        else:
            print("ERROR - sample temp folder does not exist.")
            exit(0)


# class OverallIntegration_positive(unittest.TestCase):
#     sampledata_dir = os.path.join(current_dir, "temp")
#     samplefalldata_dir = os.path.join(sampledata_dir, "fall")
#     samplenotfalldata_dir = os.path.join(sampledata_dir, "not_fall")

#     def setUp(self):
#         # create temp folder to store input data
#         if os.path.exists(self.sampledata_dir):
#             print("ERROR - Temp input data folder already exist. Please delete before running script")
#             exit(0)
#         os.mkdir(self.sampledata_dir)

#         # create temp falldata folder
#         if os.path.exists(self.samplefalldata_dir):
#             print("ERROR - Temp falldata folder already exist. Please delete before running script")
#             exit(0)
#         os.mkdir(self.samplefalldata_dir)

#         # create temp nonfalldata folder
#         if os.path.exists(self.samplenotfalldata_dir):
#             print("ERROR - Temp nonfalldata folder already exist. Please delete before running script")
#             exit(0)
#         os.mkdir(self.samplenotfalldata_dir)

#         # create sample thresholded data
#         temp_dir = os.path.join(self.samplefalldata_dir, "sample1.npy")
#         data = np.array([
#             [
#                 [2,0,0],
#                 [2,0,0],
#                 [1,0,0]
#             ],
#             [
#                 [2,0,0],
#                 [2,0,0],
#                 [1,0,0]
#             ]
#         ])
#         np.save(temp_dir, data)
#         temp_dir = os.path.join(self.samplefalldata_dir, "sample2.npy")
#         data = np.array([
#             [
#                 [2,0,0],
#                 [2,0,0],
#                 [1,0,0]
#             ],
#             [
#                 [1,0,0],
#                 [1,0,0],
#                 [0,0,0]
#             ]
#         ])
#         np.save(temp_dir, data)
#         for i in range(3, 5, 1):
#             temp_dir = os.path.join(self.samplefalldata_dir, "sample{0}.npy".format(i))
#             np.save(temp_dir, data)

#         temp_dir = os.path.join(self.samplenotfalldata_dir, "sample1.npy")
#         data = np.array([
#             [
#                 [2,0,0],
#                 [2,0,0],
#                 [1,0,0]
#             ],
#             [
#                 [2,0,0],
#                 [2,0,0],
#                 [1,0,0]
#             ]
#         ])
#         np.save(temp_dir, data)
#         temp_dir = os.path.join(self.samplenotfalldata_dir, "sample2.npy")
#         data = np.array([
#             [
#                 [2,0,0],
#                 [2,0,0],
#                 [1,0,0]
#             ],
#             [
#                 [1,0,0],
#                 [1,0,0],
#                 [0,0,0]
#             ]
#         ])
#         np.save(temp_dir, data)
#         for i in range(3, 5, 1):
#             temp_dir = os.path.join(self.samplenotfalldata_dir, "sample{0}.npy".format(i))
#             np.save(temp_dir, data)

    
#     def test_main(self):
#         program_dir = os.path.join(current_dir, "../retrieve_coordinates.py")
#         input_fall_dir = self.samplefalldata_dir
#         input_notfall_dir = self.samplenotfalldata_dir
#         output_fall_dir = os.path.join(self.sampledata_dir, "output_fall.npy")
#         output_notfall_dir = os.path.join(self.sampledata_dir, "output_notfall.npy")
#         number_of_parallel_processes = 10
#         subprocess.run("python3 {0} --input_fall_dir {1} --input_notfall_dir {2} --output_fall_dir {3} --output_notfall_dir {4} --number_of_parallel_processes {5}".format(program_dir, input_fall_dir, input_notfall_dir, output_fall_dir, output_notfall_dir, number_of_parallel_processes), shell=True)

#         fall_output = np.load(output_fall_dir, allow_pickle=True)
#         notfall_output = np.load(output_notfall_dir, allow_pickle=True)

#         to_compare_fall = np.array([
#             [
#                 'sample1.npy', 
#                 [[0, 0], [0, 1]],
#                 [[0, 0], [0, 1]]
#             ],
#             [
#                 'sample3.npy', 
#                 [[0, 0], [0, 1]],
#                 [[0, 0], [0, 1]]
#             ],
#             [
#                 'sample4.npy', 
#                 [[0, 0], [0, 1]],
#                 [[0, 0], [0, 1]]
#             ],
#             [
#                 'sample2.npy', 
#                 [[0, 0], [0, 1]],
#                 [[0, 0], [0, 1]]
#             ]
#         ])
#         to_compare_notfall = np.array([
#             [
#                 'sample1.npy',
#                 [[0, 0], [0, 1]],
#                 [[0, 0], [0, 1]]
#             ],
#             [
#                 'sample3.npy', 
#                 [[0, 0], [0, 1]],
#                 [[0, 0], [0, 1]]
#             ],
#             [
#                 'sample4.npy', 
#                 [[0, 0], [0, 1]],
#                 [[0, 0], [0, 1]]
#             ],
#             [
#                 'sample2.npy', 
#                 [[0, 0], [0, 1]],
#                 [[0, 0], [0, 1]]
#             ]
#         ])

#         self.assertTrue((fall_output==to_compare_fall).all())
#         self.assertTrue((notfall_output==to_compare_notfall).all())
    
#     def tearDown(self):
#         pass
#         # delete temp folder and contents
#         if os.path.exists(self.sampledata_dir):
#             subprocess.run("rm -rf {0}".format(self.sampledata_dir), shell=True)
#         else:
#             print("ERROR - sample temp folder does not exist.")
#             exit(0)


if __name__ == "__main__":
    unittest.main()