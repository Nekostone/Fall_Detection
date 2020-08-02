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




class Serial_Reader():
    def __init__(self):
        # Radar Ports
        self.dataport = None
        self.cfgport = None

        self.buffer = []

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


    def get_serial_data(self):
        magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        self.buffer.clear()
        byte_counter = 0

        start = time.time()
        while byte_counter < 2 * (512 * 64 + 64): # (512 * 64 + 64) is 1 packet length, getting at least 2 packets guarantees at least one complete packet
            waiting = self.dataport.in_waiting
            self.buffer.append(self.dataport.read(waiting))
            byte_counter += waiting
            # time.sleep(0.1)
        print("byte_counter: {0}".format(byte_counter))
        end = time.time()
        print("data collect: {0}".format(end-start))

        self.buffer = b''.join(self.buffer)   # combines into a single byte string
        self.buffer = self.buffer.split(magic_word) # splits into packets based on magic word at start of header
        return self.buffer

class Fall_Detector():
    def __init__(self):
        self.ml_frames = []
        self.frame_energies = []

        self.segment_length = 20
        self.ml_length = 5
        self.counter = 0

        self.data_buffer = None

        self.energy_threshold = 100
        # self.energy_threshold_2 = 
        # load svm model weights
        with open(svm_weights, "rb") as readfile:
            self.model = pickle.loads(readfile.read())

        # for logging
        self.start_script_time = datetime.datetime.now()
        self.false_positive_count = 0

        # for mqtt
        import paho.mqtt.client as mqtt
        self.client = mqtt.Client()
        self.client.connect("192.168.2.109")


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

    def process_data(self, serial_list):
        if self.data_buffer == None:
            self.parse_complete_frame(serial_list[1])
            self.data_buffer = serial_list[2]

        else:
            combined = self.data_buffer + serial_list[0]
            if len(combined) == 32824:
                self.parse_complete_frame(combined)
            else:
                print("frame skipped 0")
            self.parse_complete_frame(serial_list[1])
            self.data_buffer = serial_list[2]

    def parse_complete_frame(self, frame_string): # takes a byte string containing the whole frame excluding magic word
        print("self.start_script_time: {0}, self.false_positive_count: {1}".format(self.start_script_time, self.false_positive_count))
        ping = time.time()
        frame = frame_string[36:-20] # extract frame data
        # converts bytestring into pixel data, each element represents a pixel
        data_arr = [int.from_bytes(frame[i:i+2], byteorder = "little", signed = False) for i in range(0, len(frame), 2)] # convert to int
        if len(data_arr) != 256*64:
            print("frame skipped 1")
            return 1
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
            print("@@@@@@@  self.frame_energies[0]: {0}".format(self.frame_energies[0]))
            if output == 1:
                if np.sum(self.frame_energies[5:20])/15 >= self.energy_threshold:
                    output = 0

                if self.frame_energies[0] <= self.energy_threshold:
                    output = 0

            if output == 1:
                self.false_positive_count += 1

            # publish results to mqtt
            payload = "date: {0}; time: {1}; label: {2}".format(datetime.datetime.now().date(), datetime.datetime.now().time(), output)
            self.client.publish("fall", payload)

            pong = time.time()
            print("Current output: {0} in {1}s".format(output, pong - ping))
            self.ml_frames.pop(0)   #remove oldest frame
            self.frame_energies.pop(0)

if __name__ == "__main__":
    serial_handle = Serial_Reader()
    sentry_handle = Fall_Detector()

    serial_handle.connect_serial()
    while True:
        start = time.time()
        list_of_frames = serial_handle.get_serial_data()
        print("len(list_of_frames[-1]): {0}".format(len(list_of_frames[-1])))
        if len(list_of_frames) != 3:
            print("len(list_of_frames): {0}".format(len(list_of_frames)))
            continue
        sentry_handle.process_data(list_of_frames)
        end = time.time()
        print("function duration: {0}".format(end-start))

