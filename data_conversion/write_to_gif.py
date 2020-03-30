import numpy as np
import os, struct, csv
import seaborn as sb
from scipy.fftpack import fft, fftshift
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

os.chdir(r"C:\Users\XC199\Desktop\Crapstone\March_27_Data")

x = loadmat('20200327_JD_2m_Backwards.mat') # matrix is loaded as a dict


RadarState = x['RadarState']
ProfileConfig = RadarState[0][0][2]
FrameConfig   = RadarState[0][0][34]

print("Reading adc_data.bin")

with open('20200327_JD_2m_Backwards.bin', mode='rb') as file:
    fileContent = file.read()


# Settings
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
rangeRes = c/(2*bandwidth)
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
    print("Byte-splitting...")
    for i in range(len(filehold)//2):
        file.append(filehold[i*2:(i+1)*2])
    print("Done")

    print("Converting...")
    file = (struct.unpack('h', s) for s in file)
    print("Done")

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
        # print("List Len: " + str(len(channel_dict["rx" + str(i)])))
        # with open('channel' + str(i) + '.csv', 'a') as csvFile:
        #     writer = csv.writer(csvFile)
        #     writer.writerow(channel_dict["rx" + str(i)])
        # csvFile.close()
        channel_dict["rx" + str(i)] = np.reshape(channel_dict["rx" + str(i)], (int(len(channel_dict["rx" + str(i)])/numSamplesPerChirp), numSamplesPerChirp))
        # if i == 1:
        #     print(channel_dict["rx" + str(i)])
        print(np.shape(channel_dict["rx" + str(i)]))
        
        # Range FFT
        for index,val in enumerate(channel_dict["rx" + str(i)]):
            # print(np.shape(val))
            channel_dict["rx" + str(i)][index] = np.asarray(fft(val))
            output_dict["rx" + str(i)].append(channel_dict["rx" + str(i)][index][0:128]) # take the values at positive frequencies
        output_dict["rx" + str(i)] = np.transpose(output_dict["rx" + str(i)])

        # Doppler FFT
        for index,val in enumerate(output_dict["rx" + str(i)]):
            output_dict["rx" + str(i)][index] = fftshift(fft(val))
        output_dict["rx" + str(i)] = np.log10(np.abs(np.transpose(output_dict["rx" + str(i)])))
        print("Channel " + str(i) + " complete")
    print("Done")

    minSNR = 0
    maxSNR = maxSNR

    # plotting
    fig, ax = plt.subplots(figsize=(10,5))

    ax.set(xlabel='Range', ylabel='Doppler', title='Range-Doppler plot')

    range_doppler = sb.heatmap(output_dict["rx1"], cmap='coolwarm', vmin = minSNR, vmax = maxSNR)

    ax.set_xticks(np.arange(0, maxRange, rangeRes))
    ax.set_yticks(np.arange(-maxVel, maxVel, velRes))

    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    giflist.append(image)

    x += 1

kwargs_write = {'fps':15.0, 'quantizer':'nq'}

# giffing
imageio.mimsave('./rangedoppler1.gif', giflist, fps=15)
