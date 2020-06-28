#! usr/bin/env/python

"""
Please run this script from repo's root dir (some imports are relative to the repo's root dir*)
"""
import sys
import numpy as np
import scipy as sp
import serial
import time
from datetime import datetime
import pickle

from serial.tools import list_ports

# Pyside2 imports
from PySide2.QtWidgets import QWidget, QPushButton, QComboBox, QGridLayout, QApplication, QLabel, QTextEdit
from PySide2.QtGui import QPixmap, QImage, QFont, QColor
from PySide2.QtCore import QRunnable, Signal, Slot, QThreadPool
from PySide2.QtMultimedia import QCamera, QCameraInfo, QCameraViewfinderSettings
from PySide2.QtMultimediaWidgets import QCameraViewfinder

#Change all paths to your own when using this
# SVM preprocessing and feature extractoin layers
sys.path.append(r"D:\Documents\SUTD\Capstone\Fall_Detection\ml_final")
from ml_final.preprocess_actualdata import preprocess  # * <- leik dis wan

cfg_file = r"D:\Downloads\Telegram Desktop\profile_heat_map.cfg" 
svm_weights = r"D:\Downloads\Telegram Desktop\weights (6).pickle"
output_folder = r"D:\Documents\SUTD\Capstone\Data"

class COM_Ports(QComboBox):
    def __init__(self):
        super().__init__()
        self.get_ports()
        self.setMaximumSize(100,40)
        self.baudrate = 921000

    def get_ports(self):
        ports = list(list_ports.comports())
        for p in ports:
            self.insertItem(0,str(p))

class Logging_Window(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setCurrentFont(QFont("Comic Sans",10))
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        
class Webcam(QCamera):
    def __init__(self, cam):
        super().__init__()
        self.toggle_switch = 0 # 0: off, 1: on

        self.viewfinder = QCameraViewfinder()
        self.viewfinder.show()

        self.settings = QCameraViewfinderSettings()
        self.settings.setResolution(640,480)

        self.setViewfinder(self.viewfinder)
        self.setCaptureMode(QCamera.CaptureViewfinder)
        self.setViewfinderSettings(self.settings)

    @Slot()
    def switch(self):
        if self.toggle_switch == 0:
            self.toggle_switch = 1
            self.start()

        else:
            self.toggle_switch = 0
            self.stop()

class Radar_Plot(QLabel):
    sig = Signal(int)
    def __init__(self):
        super().__init__()
        self.data_buffer = None
        self.data = np.zeros((128,64), np.uint16)
        self.img = QImage(self.data, 128, 64, QImage.Format_Grayscale16)
        self.setPixmap(QPixmap(self.img).scaled(640,640)) 
        self.setMargin(40)

        self.counter = 4
        self.centroid = None
        #self.counter2 = 0 # for data purposes
        self.ml_frames = []

        # load svm model weights
        with open(svm_weights, "rb") as readfile:
            self.model = pickle.loads(readfile.read())

    def parse_complete_frame(self, frame_string): # takes a byte string containing the whole frame excluding magic word
        frame = frame_string[36:-20] # extract frame data
        data_arr = [int.from_bytes(frame[i:i+2], byteorder = "little", signed = False) for i in range(0, len(frame), 2)] # convert to int
        data_arr = np.asarray(data_arr).reshape((256,64))[0:128]

        # get range doppler (TI mmwave demo algo)
        #data_arr = np.add(data_arr[0::2,] , data_arr[1::2,] * 256)

        # fftshift
        data_arr = np.concatenate((data_arr[:,32:64],data_arr[:,0:32]), axis=1)
        #data_arr = np.where(data_arr == 0, 0.001, data_arr)
        #data_arr = np.log10(data_arr)

        # Store frames, yes it's not memory efficient but heck lmao
        if len(self.ml_frames) < self.counter:
            self.ml_frames.append(data_arr)
        
        # send for svm
        else:
            self.ml_frames.append(data_arr)

            #For saving data 
            # d = np.asarray(self.ml_frames)
            # d = np.moveaxis(d, 1,-1)
            # print(d.shape)
            # if self.counter2 < 100:
            #     print(self.counter2)
            #     output_path = output_folder + r"\tm_nonfall{}.npy".format(self.counter2)
            #     np.save(output_path, d)
            #     self.counter2 += 1
            # self.ml_frames = []

            # svm code here
            preprocessed = preprocess(self.ml_frames)
            output = self.model.predict(preprocessed)
            self.sig.emit(output[0]) # pass to main app

            self.ml_frames.pop(0)   #remove oldest frame


        # plot
        #data_arr = data_arr - np.min(data_arr) #invert colors 
        data_arr = (data_arr - np.min(data_arr))/(np.max(data_arr) - np.min(data_arr))
        data_arr = (255* data_arr).astype(np.uint8)
        self.data = data_arr
        self.img = QImage(self.data, 64, 128, QImage.Format_Grayscale8)
        self.setPixmap(QPixmap(self.img).scaled(640 ,640))
        self.repaint()


    @Slot(list)
    def update_image(self, serial_list): #takes in a list containing range doppler data
        if self.data_buffer == None:
            self.parse_complete_frame(serial_list[1])
            self.data_buffer = serial_list[2]

        else:
            combined = self.data_buffer + serial_list[0]
            if len(combined) == 32824:
                self.parse_complete_frame(combined)

            self.parse_complete_frame(serial_list[1])
            self.data_buffer = serial_list[2]

  
class Worker(QRunnable):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn

    @Slot()
    def run(self):
        self.fn()

class Main_Window(QWidget):
    sig = Signal(list)
    def __init__(self):
        super().__init__()
        self.available_cams = QCameraInfo.availableCameras()
        self.dataport = None
        self.cfgport = None
        self.stop = 0

        # set up qthreads
        self.threadpool = QThreadPool()

        self.layout = QGridLayout()
        self.setFixedSize(1920,960)

        #declare widgets
        self.camera = Webcam(self.available_cams[0])

        self.cfg_port = COM_Ports()
        self.data_port = COM_Ports()
        self.plot = Radar_Plot() 
        self.log = Logging_Window()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.camera_button = QPushButton("Toggle Camera")

        # declare and set up labels
        self.cfg_label = QLabel("Config")
        self.data_label = QLabel("Data")
        self.cfg_label.setMaximumSize(100,30)
        self.data_label.setMaximumSize(100,30)

        #set button callbacks
        self.start_button.clicked.connect(self.start_callback)
        self.stop_button.clicked.connect(self.stop_callback)
        self.camera_button.clicked.connect(self.camera.switch)

        #insert widgets to layout
        self.layout.addWidget(self.cfg_label, 0, 2)
        self.layout.addWidget(self.data_label, 0, 3)
        self.layout.addWidget(self.cfg_port, 1, 2)
        self.layout.addWidget(self.data_port, 1, 3)
        self.layout.addWidget(self.start_button, 2, 2)
        self.layout.addWidget(self.stop_button, 2, 3)
        self.layout.addWidget(self.camera_button, 3, 2, 1, 2)
        self.layout.addWidget(self.log, 4, 2, 4, 2)

        self.layout.addWidget(self.plot, 0, 0, 6, 1)
        self.layout.addWidget(self.camera.viewfinder, 0, 1, 6, 1)

        # add to widget
        self.setLayout(self.layout)
        self.sig.connect(self.plot.update_image) 
        self.plot.sig.connect(self.log_ml_output)      

    def start_callback(self):
        self.dataport = serial.Serial(port = self.data_port.currentText()[-5:-1], baudrate=921600)
        self.cfgport = serial.Serial(port= self.cfg_port.currentText()[-5:-1], baudrate=115200)

        self.log.append("Data port:" + self.data_port.currentText()[-5:-1] + "baudrate: 921600")
        self.log.append("Cfg port:" + self.cfg_port.currentText()[-5:-1] + "baudrate: 115200")

        f = open(cfg_file, "r")
        settings = f.readlines()
        for setting in settings:
            self.cfgport.write(setting.encode())
            self.log.append(setting)
            time.sleep(0.05)

        time.sleep(0.2)

        # start getting serial data and doing ml
        serial_worker = Worker(self.get_serial_data)  
        self.threadpool.start(serial_worker)


    def stop_callback(self):
        try:
            self.stop = 1
            self.cfgport.write(("sensorStop\n").encode())
            self.log.append("sensorStop")
            self.cfgport.close()
            self.dataport.close()

        except:
            self.log.append("Invalid port")

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
                time.sleep(0.05)

            buffer = b''.join(buffer)   # combines into a single byte string
            buffer = buffer.split(magic_word) # splits into packets based on magic word at start of header

            self.sig.emit(buffer)
                
    @Slot(int)
    def log_ml_output(self,ml_out):
        if ml_out == 1:
            t = datetime.now()
            current_time_h = t.hour
            current_time_min = t.minute
            current_time_s = t.second
            current_time_ms = t.microsecond/1000
            self.log.append("{0}:{1}:{2}:{3} is fall".format(current_time_h, current_time_min, current_time_s, current_time_ms))


if __name__ == "__main__":
    app = QApplication([])
    widget = Main_Window()
    widget.setWindowTitle("Viewer")
    widget.show()

    sys.exit(app.exec_())






    

