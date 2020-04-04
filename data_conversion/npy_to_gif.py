"""
TO USE THIS PROGRAM:

Run
```python3 ./npy_to_gif.py --input_folder {input_folder_dir} --input_filename {input_filename} --output_folder {output_folder_dir}```

Where {input_folder_dir} is the absolute folder directory of the particular file,
    {input_filename} is the filename of the numpy array without .npy
    and {output_folder_dir} is the absolute folder directory of where the output should go
"""

import numpy as np
import os
import seaborn as sb
from matplotlib import pyplot as plt
import io
import copy
import PIL.Image as Image
import argparse

def numpy_to_gif(input_folder, input_file,output_folder):
    filename = input_file
    giflist = []
    os.chdir(input_folder)
    data = np.load(filename+".npy")
    #print(data[0].shape)
    minSNR = 0
    maxSNR = 6
    # fig, ax = plt.subplots(figsize=(10,5))
    giflist = []
    # ax.set(xlabel='Range', ylabel='Doppler', title='Range-Doppler plot')
    x=0
    for i in data:
        fig = plt.figure()
        #Generate image for one plot 
        plt.xlabel('Range',fontsize=20)
        plt.ylabel("Doppler",fontsize=20)
        plt.title('Range-Doppler plot',fontsize=20)
        range_doppler = sb.heatmap(i, cmap='coolwarm', vmin = minSNR, vmax = maxSNR)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi = 50)
        buf.seek(0)
        pil_img = copy.deepcopy(Image.open(buf))
        #add image to giflist which will contain the entire set of images
        giflist.append(pil_img)
        buf.close()
        plt.close()
        x+=1
        print(x)
    os.chdir(output_folder)
    giflist[0].save(filename +'.gif', format="GIF",append_images=giflist[1:],save_all=True,duration=50, loop=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieves absolute directories for input and output file names')
    parser.add_argument('--input_folder', dest='input_folder', help="folder containing input data")
    parser.add_argument('--input_filename', dest='input_filename', help="shows output")
    parser.add_argument('--output_folder', dest='output_folder', help="shows output")
    args = parser.parse_args()

    if args.input_filename and args.output_folder and args.input_folder:
        # print("args.input_filename[0]: ", args.input_filename)
        # print("args.output_filename[0]: ", args.output_filename)
        numpy_to_gif(args.input_folder,args.input_filename, args.output_folder)
    else:
        print("Please input your input folder/output folder/input filename!")
