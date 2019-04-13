import os, sys
import imageio
from tqdm import tqdm
import numpy as np

path = 'output/Gaussian_grid'
iter_ = np.arange(0,20000,200)
fname_output = 'output_grid.gif'

img_list = []
for idx in tqdm(iter_):
    fname = os.path.join(path, 'img_grid', 'img_{}.png'.format(idx))
    img = imageio.imread(fname)
    img_list.append(img)

imageio.mimsave(fname_output, img_list, duration=0.1)
