import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
from os import listdir
from os import chdir
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

path = os.getcwd()+'/output'

def image_path(file):
	global path
	return path+'/'+file


files = listdir(path)
images = [ image_path(f) for f in files]

num_rows = 2
num_cols = 4

fig = plt.figure()
gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.0)

ax = [plt.subplot(gs[i]) for i in range(num_rows*num_cols)]
gs.update(hspace=0)
#gs.tight_layout(fig, h_pad=0,w_pad=0)

for i,im in enumerate(images):
	image=mpimg.imread(im)
	ax[i].imshow(image)
	ax[i].axis('off')

plt.show()