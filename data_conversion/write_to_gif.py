import numpy as np
import os, struct, csv
import seaborn as sb
from scipy.fftpack import fft, fftshift
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import scipy.io as sio
from matplotlib import animation
import io
import copy
import PIL.Image as Image
import argparse


#Construct arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-folder", "--folderpath",required=True,help="folder path")
ap.add_argument("-file","--filename", required=True,help="file name (e.g. 20200320_ttt_4m_sideways)")
args = vars(ap.parse_args())
folderpath = str(args['folderpath'])
os.chdir(folderpath.encode('unicode_escape'))
filename = str(args['filename'])

x = sio.loadmat(filename+".mat") # matrix is loaded as a dict


RadarState = x['RadarState']
ProfileConfig = RadarState[0][0][2]
FrameConfig   = RadarState[0][0][-1]

print("Reading bin file")

with open(filename+".bin", mode='rb') as file:
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

x = 0 #just an iterable to go through all the data stored in file

while (x < len(fileContent)//chirpDataLen):
    filehold = fileContent[x*chirpDataLen:(x+1)*chirpDataLen]
    file = []

    # split byte inputs into 2-byte packets
    # print("Byte-splitting...")
    for i in range(len(filehold)//2):
        file.append(filehold[i*2:(i+1)*2])
    # print("Done")

    # print("Converting...")
    file = (struct.unpack('h', s) for s in file)
    # print("Done")

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
    print("Performing range-doppler conversion for each rx")
    for i in range(4):
        channel_dict["rx" + str(i)] = np.asarray([complex(s, y) for s, y in zip(list_dict["rx" + str(i) + "_real"], list_dict["rx" + str(i) + "_comp"])])
        channel_dict["rx" + str(i)] = np.reshape(channel_dict["rx" + str(i)], (int(len(channel_dict["rx" + str(i)])/numSamplesPerChirp), numSamplesPerChirp))
        # print(np.shape(channel_dict["rx" + str(i)]))
        
        # Range FFT
        for index,val in enumerate(channel_dict["rx" + str(i)]):
            channel_dict["rx" + str(i)][index] = np.asarray(fft(val))
            output_dict["rx" + str(i)].append(channel_dict["rx" + str(i)][index][0:128]) # take the values at positive frequencies
        output_dict["rx" + str(i)] = np.transpose(output_dict["rx" + str(i)])

        # Doppler FFT
        for index,val in enumerate(output_dict["rx" + str(i)]):
            output_dict["rx" + str(i)][index] = fftshift(fft(val))
        output_dict["rx" + str(i)] = np.log10(np.abs(np.transpose(output_dict["rx" + str(i)])))
    #     print("Channel " + str(i) + " complete")
    # print("Done")
b
    minSNR = 0
    maxSNR = maxSNR

    # plotting
    fig = plt.figure()
    plt.xlabel('Range',fontsize=20)
    plt.ylabel("Doppler",fontsize=20)
    plt.title('Range-Doppler plot',fontsize=20)
    print(output_dict["rx1"])
    range_doppler = sb.heatmap(output_dict["rx1"], cmap='coolwarm', vmin = minSNR, vmax = maxSNR)
    #range_doppler.swap_axes()
    fig.canvas.draw()
    #convert to PIL image object and then append the created Image into giflist
    #Set dpi to desired level(Higher means bigger and more costly in terms of memory)
    buf = io.BytesIO()
    fig.savefig(buf,format="png",dpi=50)
    buf.seek(0)
    pil_img = copy.deepcopy(Image.open(buf))
    giflist.append(pil_img)
    buf.close()

    x += 1
    print(x)



gif file created. gif file will be saved with same name as file
giflist[0].save('./'+ args['filename']+'.gif', format="GIF",append_images=giflist[1:],save_all=True,duration=50, loop=0)

plotting


kwargs_write = {'fps':15.0, 'quantizer':'nq'}

# giffing
imageio.mimsave('./rangedoppler1.gif', giflist, fps=15)