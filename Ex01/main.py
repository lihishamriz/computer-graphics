import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from PIL import Image
from numba import jit
from utils import SeamImage, ColumnSeamImage, VerticalSeamImage

plt.rcParams["figure.figsize"] = (10,5)

# img_path = "pinguins.jpg"
img_path = "sunset.jpg"

# helper functions
def read_image(img_path):
    return np.asarray(Image.open(img_path)).astype('float32')


def show_image(np_img, grayscale=False):
    fig, ax = plt.subplots()
    if not grayscale:
        ax.imshow(np_img)
    else:
        ax.imshow(np_img, cmap=plt.get_cmap('gray'))
    ax.axis("off")
    plt.show()

def init_plt_grid(nrow=1, ncols=1, figsize=(20,10), **kwargs):
    fig, ax = plt.subplots(nrow, ncols, figsize=figsize, facecolor='gray', **kwargs)
    font_size = dict(size=20)
    return ax, font_size

s_img = SeamImage(img_path)
s_img = ColumnSeamImage(img_path)
# s_img.seams_removal_vertical(2)
# s_img.seams_removal_horizontal(2)

vs_img = VerticalSeamImage(img_path)
vs_img.seams_removal_vertical(1)

