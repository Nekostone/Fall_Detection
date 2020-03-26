"""
TO USE THIS PROGRAM:

Run
```python3 ./read_raw_bin.py {input_file_dir} {output_file_dir}```

Where {input_file_dir} is the absolute directory of the input binary file 
    and {output_file_dir} is the absolute directory of the output binary file

data format
(bredth, length, rgb)


"""

import numpy as np
import os, struct, csv
import seaborn as sb
from scipy.fftpack import fft, fftshift
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import argparse



def main(input_file_dir, output_file_dir):
    with open(input_file_dir, mode='rb') as file:
        fileContent = file.read()

    # Settings
    numSamplesPerChirp = 256
    numChirpLoops      = 128
    numChannels        = 4
    numFrames          = 8
    byteLength         = 2
    chirpDataLen       = numSamplesPerChirp*2*byteLength*numChirpLoops*numChannels
    startFreq          = 77*(10**9)

    # Constant
    c        = 3*(10**8)
    slope    = (64.985*10**12)
    rampTime = 60*10**-6
    bandwidth = slope*rampTime
    framePeriod = 40*(10**-3)

    # Axis Calculation
    rangeRes = c/(2*bandwidth)
    maxRange = rangeRes*numSamplesPerChirp/2
    velRes = (c/startFreq)/(2*framePeriod)
    maxVel = (numChirpLoops/2)*velRes

    maxSNR = 6

    # giflist = []
    state_isfirst = True
    data = None

    x = 0 #just an iterable to go through all the data stored in file

    while (x < len(fileContent)//chirpDataLen):
        filehold = fileContent[x*chirpDataLen:(x+1)*chirpDataLen]
        file = []

        # split byte inputs into 2-byte packets
        for i in range(len(filehold)//2):
            file.append(filehold[i*2:(i+1)*2])

        # print("Converting...")
        file = (struct.unpack('h', s) for s in file)

        # Convert to int
        file = [int(s[0]) for s in file]

        list_dict = {}
        channel_dict = {}
        output_dict = {}

        # write storage lists in dict
        for i in range(4):
            list_dict["rx" + str(i) + "_real"] = []
            list_dict["rx" + str(i) + "_comp"] = []
            channel_dict["rx" + str(i)] = []
            output_dict["rx" + str(i)] = []

        for i in range(len(file)):
            if (i//4)%2 == 0: # real values
                list_dict['rx' + str(i%4) + "_real"].append(file[i])
            else:
                list_dict['rx' + str(i%4) + "_comp"].append(file[i])

        # arrange all data into channel dictionary and reshape to size r-d plot size
        for i in range(4):
            channel_dict["rx" + str(i)] = np.asarray([complex(s, y) for s, y in zip(list_dict["rx" + str(i) + "_real"], list_dict["rx" + str(i) + "_comp"])])
            channel_dict["rx" + str(i)] = np.reshape(channel_dict["rx" + str(i)], (int(len(channel_dict["rx" + str(i)])/numSamplesPerChirp), numSamplesPerChirp))
            
            # Range FFT
            for index,val in enumerate(channel_dict["rx" + str(i)]):
                channel_dict["rx" + str(i)][index] = np.asarray(fft(val))
                output_dict["rx" + str(i)].append(channel_dict["rx" + str(i)][index][0:128]) # take the values at positive frequencies
            output_dict["rx" + str(i)] = np.transpose(output_dict["rx" + str(i)])

            # Doppler FFT
            for index,val in enumerate(output_dict["rx" + str(i)]):
                output_dict["rx" + str(i)][index] = fftshift(fft(val))
            output_dict["rx" + str(i)] = np.log10(np.abs(np.transpose(output_dict["rx" + str(i)])))

        lengthx = output_dict["rx1"].shape[0]
        lengthy = output_dict["rx1"].shape[1]
        if state_isfirst:
            state_isfirst = False
            output_dict["rx1"].resize((1, lengthx, lengthy))
            data = output_dict["rx1"]
        else:
            output_dict["rx1"].resize((1, lengthx, lengthy))
            data = np.append(data, output_dict["rx1"], axis=0)

        x += 1

    np.save(output_file_dir, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieves absolute directories for input and output file names')
    parser.add_argument('--input_filename', dest='input_filename', help="shows output")
    parser.add_argument('--output_filename', dest='output_filename', help="shows output")
    args = parser.parse_args()

    if args.input_filename and args.output_filename:
        # print("args.input_filename[0]: ", args.input_filename)
        # print("args.output_filename[0]: ", args.output_filename)
        main(args.input_filename, args.output_filename)
    else:
        main("/home/xubuntu/Desktop/sensor_data/raw/fall/202030_ttt_3m_radial2.bin", "/home/xubuntu/Desktop/sensor_data/processed/fall/202030_ttt_3m_radial2.npy")
