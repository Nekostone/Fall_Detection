"""
TO USE THIS PROGRAM:

Run
```python3 ./read_raw_bin.py --input_folder {input_folder_dir} --input_filename {input_filename} --output_folder {output_folder_dir}```

Where {input_folder_dir} is the absolute folder directory of the particular file,
    {input_filename} is the filename without the filetype indicated at the back
    and {output_folder_dir} is the absolute folder directory of where the output should go

data format
(bredth, length, rgb)


"""
#Input folder path, input filename(without file type) and output folder required
#output npy array name will be same as input file name

import numpy as np
import os, struct, csv
import seaborn as sb
from scipy.fftpack import fft, fftshift
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import argparse
import scipy.io as sio
import copy
import PIL.Image as Image
import io


def main(input_folder, input_filename, output_folder):
    #Change directory to input_folder
    os.chdir(input_folder.encode('unicode_escape'))
    try:
        x = sio.loadmat(input_filename+".mat") # matrix is loaded as a dict
    except:
        print("%s has no .mat file :(. Proceed to use a 20200320_ttt_4m_sideways5's mat".format(input_filename))
        x = sio.loadmat("20200320_ttt_4m_sideways5.mat")

    RadarState = x['RadarState']
    ProfileConfig = RadarState[0][0][2]
    FrameConfig   = RadarState[0][0][-1]

    with open(input_filename + ".bin", mode='rb') as file:
        fileContent = file.read()


    # Settings taken from .mat file
    numSamplesPerChirp = ProfileConfig[0][0][9][0][0]
    numChirpLoops      = FrameConfig[0][0][0][0][0][0][0][0][3][0][0]
    numChirpConfs      = FrameConfig[0][0][0][0][0][0][0][0][2][0][0]
    numChirpLoops      = numChirpLoops*numChirpConfs
    numFrames          = FrameConfig[0][0][3][0][0]

    numTxAnt           = 1
    numChannels        = numTxAnt*4

    byteLength         = 2
    chirpDataLen       = numSamplesPerChirp*2*byteLength*numChirpLoops*numChannels
    startFreq          = 77*(10**9)

    # Constant
    c           = 3*(10**8)
    slope       = ProfileConfig[0][0][7][0][0]
    rampTime    = ProfileConfig[0][0][4][0][0]
    bw          = slope*rampTime
    startFreq   = ProfileConfig[0][0][1][0][0]
    framePeriod = FrameConfig[0][0][1][0][0]

    # Axis Calculation
    rangeRes = c/(2*bw)
    maxRange = rangeRes*numSamplesPerChirp/2
    velRes = (c/startFreq)/(2*framePeriod)
    maxVel = (numChirpLoops/2)*velRes

    maxSNR = 6

    giflist = []
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
        # fig = plt.figure()
        # #Generate image for one plot 
        # plt.xlabel('Range',fontsize=20)
        # plt.ylabel("Doppler",fontsize=20)
        # plt.title('Range-Doppler plot',fontsize=20)
        # minSNR = 0
        # range_doppler = sb.heatmap(output_dict["rx1"], cmap='coolwarm', vmin = minSNR, vmax = maxSNR)
        # fig.canvas.draw()
        # #convert to PIL image object and then append the created Image into giflist
        # #Set dpi to desired level(Higher means bigger and more costly in terms of memory)
        # buf = io.BytesIO()
        # fig.savefig(buf,format="png",dpi=50)
        # buf.seek(0)
        # pil_img = copy.deepcopy(Image.open(buf))
        # #add image to giflist which will contain the entire set of images
        # giflist.append(pil_img)
        # buf.close()
        if state_isfirst:
            state_isfirst = False
            output_dict["rx1"].resize((1, lengthx, lengthy))
            #output_dict["rx1"] = output_dict["rx1"].resize((1, lengthx, lengthy))
            data = output_dict["rx1"]
        else:
            output_dict["rx1"].resize((1, lengthx, lengthy))
            #output_dict["rx1"] = output_dict["rx1"].resize((1, lengthx, lengthy))
            data = np.append(data, output_dict["rx1"], axis=0)
        print(x)
        x += 1
    os.chdir(output_folder.encode('unicode_escape'))
    np.save(input_filename, data)
    #giflist[0].save(input_filename+'.gif', format="GIF",append_images=giflist[1:],save_all=True,duration=50, loop=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieves absolute directories for input and output file names')
    parser.add_argument('--input_folder', dest='input_folder', help="folder containing input data")
    parser.add_argument('--input_filename', dest='input_filename', help="shows output")
    parser.add_argument('--output_folder', dest='output_folder', help="shows output")
    args = parser.parse_args()

    if args.input_filename and args.output_folder and args.input_folder:
        # print("args.input_filename[0]: ", args.input_filename)
        # print("args.output_filename[0]: ", args.output_filename)
        main(args.input_folder,args.input_filename, args.output_folder)
    else:
        # print(args.input_folder)
        # print(args.input_filename)
        # print(args.output_folder)
        print("Please input your input folder/output folder/input filename!")
        #main("/home/xubuntu/Desktop/sensor_data/raw/fall/202030_ttt_3m_radial2.bin", "/home/xubuntu/Desktop/sensor_data/processed/fall/202030_ttt_3m_radial2.npy")
