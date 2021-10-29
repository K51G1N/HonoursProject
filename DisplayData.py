# ---------------------------------------------------------------------------------------------------------------
#Load your own image files with categories as subfolder names
# This example assumes that the images are preprocessed, and classifies using tuned SVM
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def load_image_files(container_path):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "Your own dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            ps = np.load(file)
            # print(ps.shape)
            # rect = patches.Rectangle((13, 13), 26, 26, linewidth=3, edgecolor='r', facecolor='none')
            fig, ax = plt.subplots()
            ax.imshow(ps)
            # ax.add_patch(rect)
            # plt.imshow(ps)
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.show()

# load_image_files("/media/keagan/Keagan HDD/Honours/PFVaryExperiment/Test/85")

pathToFolder = input('Give the absolute path to the a parent Folder of the folder containing the .npy files you\'re wanting to view: \n example: Experiment1/TrainData/Extract \n ...')
load_image_files(pathToFolder)