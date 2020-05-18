"""
Visualise npy files either as a heatmap or as a time series gif
"""
import numpy as np
import argparse
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import imageio


def show_heatmap(input_dir):
    input_array = np.load(input_dir)

    print("input_array: ", input_array)
    print("type(input_array): \n", type(input_array))
    sns.heatmap(input_array, cmap='coolwarm')
    plt.show()

def generate_gif(input_dir, output_dir):
    input_array = np.load(input_dir)
    input_array = input_array.copy()
    gif_list = []
    state_isfirst = True
    for i in range(input_array.shape[0]):
        fig, ax = plt.subplots(figsize=(10,5))

        ax.set(xlabel='Range', ylabel='Doppler', title='Range-Doppler plot')

        range_doppler = sns.heatmap(input_array[0], cmap='coolwarm')

        # ax.set_xticks(np.arange(0, maxRange, rangeRes))
        # ax.set_yticks(np.arange(-maxVel, maxVel, velRes))

        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        gif_list.append(image)

        plt.close()
    
    # print("gif_list.shape: ", gif_list.shape)
    print("len(gif_list): ", len(gif_list))
    imageio.mimsave(output_dir, gif_list, fps=15)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Applies thresholding filter on numpy array file')
    parser.add_argument('--input_filename', dest='input_filename', help="input dir of .npy data file")
    parser.add_argument('--output_filename', dest='output_filename', help="output dir of .gif file (for mode 2)")
    parser.add_argument('--mode', dest='mode', help="sets the mode of visualisation. mode 1 for heatmap, mode 2 for time series gif")
    args = parser.parse_args()

    if args.mode:
        if args.mode == "1":
            if args.input_filename:
                show_heatmap(args.input_filename)
            else:
                print("ERROR: input_filename is not specified")
        elif args.mode == "2":
            if args.input_filename and args.output_filename:
                generate_gif(args.input_filename, args.output_filename)
            else:
                print("ERROR: input_filename and/or output_filename is not specified")
    else:
        # show_heatmap("/home/xubuntu/Desktop/capstone/ml/sample1.npy", 1)
        generate_gif("/home/xubuntu/Desktop/capstone/ml/raw_data_npy/fall/20200320_jd_3m_sideways2.npy", "/home/xubuntu/Desktop/capstone/ml/20200320_jd_3m_sideways2.gif")
