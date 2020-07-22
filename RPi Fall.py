#!usr/bin/env python

import serial
import numpy as np
import pickle
import time
import datetime
import sys

from multiprocessing import Queue, Process
from scipy.signal import convolve

env = "junde"

if env == "tm":
    # files
    cfg_file = r"D:\Documents\SUTD\Capstone\Fall_Detection\profile_heat_map.cfg"
    svm_weights = r"D:\Documents\SUTD\Capstone\Ml_Data\range_rangeDelta_doppDelta_94.pickle"

    #Change all paths to your own when using this
    # SVM preprocessing and feature extractoin layers
    sys.path.append(r"D:\Documents\SUTD\Capstone\Fall_Detection\ml_final")
    from ml_final.preprocess_actualdata import preprocess  # * <- leik dis wan

    dataport_dir = "COM4"
    cfgport_dir = "COM5"
elif env == "junde":
    # files
    cfg_file = "/home/pi/Desktop/Fall_Detection/profile_heat_map.cfg"
    svm_weights = "/home/pi/Downloads/range_rangeDelta_doppDelta_94.pickle"

    #Change all paths to your own when using this
    # SVM preprocessing and feature extractoin layers
    sys.path.append("/home/pi/Desktop/Fall_Detection/ml_final")
    from ml_final.preprocess_actualdata import preprocess  # * <- leik dis wan

    dataport_dir = "/dev/ttyACM1"
    cfgport_dir = "/dev/ttyACM0"



start_script_time = datetime.datetime.now()
false_positive_count = 0

class Serial_Reader():
    def __init__(self, q):
        # multiprocessing queue reference
        self.queue = q
        self.stop = 0

        # Radar Ports
        self.dataport = None
        self.cfgport = None

    def connect_serial(self):
        self.dataport = serial.Serial(port = dataport_dir, baudrate=921600)
        self.cfgport = serial.Serial(port= cfgport_dir, baudrate=115200)

        print("Connecting to data port: {}".format(self.dataport))
        print("Connecting to config port: {}".format(self.cfgport))

        f = open(cfg_file, "r")
        print("Sending config file via serial....")
        settings = f.readlines()
        for setting in settings:
            self.cfgport.write(setting.encode())
            time.sleep(0.05)

        time.sleep(0.2)

        print("Start data acquisition...")
        while True:
            self.get_serial_data()

    def get_serial_data(self):
        magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        self.stop = 0
        while self.stop == 0:
            buffer = []
            byte_counter = 0

            while byte_counter < 2 * (512 * 64 + 64): # (512 * 64 + 64) is 1 packet length, getting at least 2 packets guarantees at least one complete packet
                waiting = self.dataport.in_waiting
                buffer.append(self.dataport.read(waiting))
                byte_counter += waiting
                # time.sleep(0.1)

            buffer = b''.join(buffer)   # combines into a single byte string
            buffer = buffer.split(magic_word) # splits into packets based on magic word at start of header
            if len(buffer) < 4:
                self.queue.put(buffer) # buffer is a list with min length 2, max length 3
                # print("Sent frame")
                print("Sent frame. start_script_time: {0}, false_positive_count: {1}".format(start_script_time, false_positive_count))
            

class Fall_Detector():
    def __init__(self, q):
        # multiprocessing queue
        self.queue = q
        self.data_buffer = None   # for incomplete frames

        self.ml_frames = []
        self.frame_energies = []

        self.segment_length = 20
        self.ml_length = 5
        self.counter = 0

        self.energy_threshold = 100
        # load svm model weights
        with open(svm_weights, "rb") as readfile:
            self.model = pickle.loads(readfile.read())


    def cfar(self,arr):
        # one side
        train = 5
        guard = 3
        p = 1.4

        train_size = 2*(train + guard) + 1
        guard_size = 2* guard +1
        arr2 = np.pad(arr,train + guard,mode="mean")

        # build kernel
        kernel = np.zeros((7,7))
        kernel = np.pad(kernel, 5, constant_values = 1)

        # perform cfar operation
        ave_noise = convolve(arr2, kernel, mode="same")[train+guard : train+guard+128, train+guard: train+guard+64] /(train_size**2 - guard_size**2)
        
        truth = np.greater_equal(arr, p* ave_noise)
        output = np.where(truth, arr, 0)

        return output

    def process_data(self):
        while True:
            if self.queue.empty() == False:
                serial_list = self.queue.get()
                if self.data_buffer == None:
                    self.parse_complete_frame(serial_list[1])
                    self.data_buffer = serial_list[2]

                else:
                    combined = self.data_buffer + serial_list[0]
                    if len(combined) == 32824:
                        self.parse_complete_frame(combined)

                    self.parse_complete_frame(serial_list[1])
                    self.data_buffer = serial_list[2]

    def parse_complete_frame(self, frame_string): # takes a byte string containing the whole frame excluding magic word
        ping = time.time()
        frame = frame_string[36:-20] # extract frame data
        data_arr = [int.from_bytes(frame[i:i+2], byteorder = "little", signed = False) for i in range(0, len(frame), 2)] # convert to int
        data_arr = np.asarray(data_arr).reshape((256,64))[0:128]/512 # data is in q9 format, so have to divide by 2**9 = 512 to get true value

        # fftshift
        data_arr = np.concatenate((data_arr[:,32:64],data_arr[:,0:32]), axis=1)

        #cfar
        data_arr = self.cfar(data_arr)
        data_arr[:,31:34] = 0 # remove centre here because energy computation requires center to be removed

        # Store frames, yes it's not memory efficient but heck lmao
        if len(self.ml_frames) < self.segment_length:
            self.ml_frames.append(data_arr)
            self.frame_energies.append(np.sum(data_arr))
        
        # send for svm
        else:
            self.ml_frames.append(data_arr)
            self.frame_energies.append(np.sum(data_arr))

            # svm code here
            preprocessed = preprocess(self.ml_frames[0:self.ml_length])
            output = self.model.predict(preprocessed)

            # perform second level check to eliminate false positives based on energy levels post fall
            if output == 1:
                if np.sum(self.frame_energies[5:20])/15 >= self.energy_threshold:
                    output = 0

                false_positive_count += 1

            pong = time.time()
            print("Current output: {0} in {1}s".format(output, pong - ping))
            self.ml_frames.pop(0)   #remove oldest frame
            self.frame_energies.pop(0)

if __name__ == "__main__":
    q = Queue()
    serial_handle = Serial_Reader(q)
    sentry_handle = Fall_Detector(q)

    serial_process = Process(target = serial_handle.connect_serial)
    serial_process.start()

    sentry_handle.process_data()
