'''
    This file is used to plot a comparison between number of copies of the
    holographic memory vs the retrieval error
'''

import math
from collections import namedtuple
from functools import reduce

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc

from holographic_memory import *

rc('font', family='serif')
sns.set(style="whitegrid")

# Plot Config
linewidth = 6
labelsize_major = 24
labelsize_minor = 18
legend_size = 24

# Plot Colors
plt_color = '#24b476'

plt.ion()
plt.cla()


if __name__ == '__main__':

  # Setting Seed
  np.random.seed(seed = 1234)

  # Loading data
  data = np.load('20_images.npy')

  img_dim = int(reduce((lambda x, y: x * y), list(data.shape)) / data.shape[0])
  data = data.reshape(data.shape[0], img_dim)
  temp_data = data
  data_mean = data.mean(axis = 1, keepdims = True)
  data -= data_mean
  data = np.append(data[:, :img_dim//2], data[:, img_dim//2:], axis = 1)

  errs = []
  cells_count = [1, 2, 5, 10, 20, 30, 50, 100]

  # Generating keys
  thetas = np.random.random((data.real.shape[0], img_dim // 2)) * 2 * np.pi
  keys = np.append(np.cos(thetas), np.sin(thetas), axis = 1)

  for i in cells_count:

    # Constructing the Holographic Memory
    # num_cells = args.num_cells
    num_cells = i
    cell_size = img_dim
    memory = HolographicMemory(num_cells, cell_size)

    memory.write(keys, data)

    images = memory.read(keys)

    # images = np.array(images)
    images += data_mean

    err = ((images - temp_data) ** 2).mean(axis=None)
    errs += [err]

  # Plotting result
  plt.plot(cells_count, errs, lw=linewidth, color=plt_color,
           label='cells vs accuracy')
  plt.xlabel('Cells', fontsize=labelsize_major)
  plt.ylabel('MSE', rotation=0,fontsize=labelsize_major)
  plt.legend(borderaxespad=0.2, bbox_to_anchor=(.005, 1.17, 1., .102),
            fontsize=legend_size, ncol=2, mode="expand",
            numpoints = 1, handlelength=1)
  plt.tick_params(axis='both', which='major', labelsize=labelsize_major)
  plt.tick_params(axis='both', which='minor', labelsize=labelsize_minor)
  plt.tight_layout()
  plt.savefig('cells-vs-accuracy.pdf', bbox_inches='tight', dpi=200)
