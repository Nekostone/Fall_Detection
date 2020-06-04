#! usr/bin/env/python

"""
Please run this script from repo's root dir (some imports are relative to the repo's root dir*)
"""
import sys
import numpy as np
import serial
import time
import pickle

from serial.tools import list_ports

# Pyside2 imports
from PySide2.QtWidgets import QWidget, QPushButton, QComboBox, QGridLayout, QApplication, QLabel, QTextEdit
from PySide2.QtGui import QPixmap, QImage, QFont, QColor
from PySide2.QtCore import QRunnable, Signal, Slot, QThreadPool

# SVM preprocessing and feature extractoin layers
sys.path.append(r"D:\Documents\SUTD\Capstone\Fall_Detection\ml_final")
from ml_final.preprocess_actualdata import preprocess  # * <- leik dis wan

cfg_file = r"D:\Downloads\Telegram Desktop\profile_heat_map.cfg"
svm_weights = r"D:\Downloads\Telegram Desktop\weights (2).pickle"

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
        

class Radar_Plot(QLabel):
    sig = Signal(list)
    def __init__(self):
        super().__init__()
        self.data_buffer = None
        self.data = np.zeros((128,64), np.uint16)
        self.img = QImage(self.data, 128, 64, QImage.Format_Grayscale16)
        self.setPixmap(QPixmap(self.img).scaled(768,768)) 
        self.setMargin(100)

        self.ml_frames = []

        # load svm model weights
        with open(svm_weights, "rb") as readfile:
            self.model = pickle.loads(readfile.read())

    def parse_complete_frame(self, frame_string): # takes a byte string containing the whole frame excluding magic word
        frame = frame_string[36:-20] # extract frame data
        data_arr = [int.from_bytes(frame[i:i+2], byteorder = "little", signed = False) for i in range(0, len(frame), 2)] # convert to int
        data_arr = np.asarray(data_arr).reshape((256,64))[0:128,]

        # get range doppler (TI mmwave demo algo)
        #data_arr = np.add(data_arr[0::2,] , data_arr[1::2,] * 256)

        # fftshift
        data_arr = np.concatenate((data_arr[:,32:64],data_arr[:,0:32]), axis=1)

        # Store frames, yes it's not memory efficient but heck lmao
        if len(self.ml_frames) == 0:
            self.ml_frames.append(data_arr)
        
        # send for svm
        else:
            self.ml_frames.append(data_arr)
            # svm code here
            preprocessed = preprocess(self.ml_frames)
            output = self.model.predict(preprocessed)
            if output == 1:
                print("Label: Is fall")
            else:
                print("Label: lolno")

            # extract features and do svm code here
            # print the answer or something, i'll find a way to hook it up to the actual gui after
            self.ml_frames.pop(0)   #remove oldest frame           

        # plot
        data_arr = 65535 - data_arr #invert colors
        data_arr = data_arr.astype(np.uint16) # divide by 64 to bring it down to within 16 bits
        self.data = data_arr
        self.img = QImage(self.data, 64, 128, QImage.Format_Grayscale16)
        self.setPixmap(QPixmap(self.img).scaled(768,768))
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
        self.dataport = None
        self.cfgport = None
        self.stop = 0

        # set up qthreads
        self.threadpool = QThreadPool()

        self.layout = QGridLayout()
        self.setFixedSize(1280,960)

        #declare widgets
        self.cfg_port = COM_Ports()
        self.data_port = COM_Ports()
        self.plot = Radar_Plot() 
        self.log = Logging_Window()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        # declare and set up labels
        self.cfg_label = QLabel("Config")
        self.data_label = QLabel("Data")
        self.cfg_label.setMaximumSize(100,30)
        self.data_label.setMaximumSize(100,30)

        #set button callbacks
        self.start_button.clicked.connect(self.start_callback)
        self.stop_button.clicked.connect(self.stop_callback)

        #insert widgets to layout
        self.layout.addWidget(self.cfg_label, 0, 1)
        self.layout.addWidget(self.data_label, 0, 2)
        self.layout.addWidget(self.cfg_port, 1, 1)
        self.layout.addWidget(self.data_port, 1, 2)
        self.layout.addWidget(self.start_button, 2, 1)
        self.layout.addWidget(self.stop_button, 2, 2)
        self.layout.addWidget(self.plot, 0, 0, 6 , 1)
        self.layout.addWidget(self.log, 3, 1, 4, 2)

        # add to widget
        self.setLayout(self.layout)

        self.sig.connect(self.plot.update_image)          


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
                
    def do_ml(self):
        output = False 
        self.log.append(str(output)) # log to output terminal


if __name__ == "__main__":
    app = QApplication([])
    widget = Main_Window()
    widget.setWindowTitle("Viewer")
    widget.show()

    sys.exit(app.exec_())






    

