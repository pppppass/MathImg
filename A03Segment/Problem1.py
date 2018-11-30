
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import skimage.io
from matplotlib import pyplot
from utils import downsamp
import gac


# In[7]:


filename_list = [
    "triangle",
    "objects",
    "cells",
    "bird",
    "lena",
    "konata",
]
alpha_list = [
    150.0,
    100.0, 
    100.0,
    100.0,
    100.0,
    100.0,
]
tau_list = [
    0.012 * (1.0 / 256)**2,
    0.02 * (1.0 / 244)**2,
    0.02 * (1.0 / 246)**2,
    0.02 * (1.0 / 160)**2,
    0.02 * (1.0 / 256)**2,
    0.02 * (1.0 / 256)**2,
]
edge_list = [
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
]
len_list = [
    (2000, 10),
    (2000, 10),
    (2000, 10),
    (2000, 10),
    (1000, 10),
    (1000, 10),
]


# In[3]:


rt = [{}]


# In[8]:


for j, filename in enumerate(filename_list):
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i = downsamp.samp_down(i)
    u = None
    iters, len_ = len_list[j]
    elapsed = 0.0
    u = gac.calc_init(i, 0.02)
    for k in range(len_):
        start = time.time()
        u = gac.evolve_gac(i, alpha_list[j], tau_list[j], edge_list[j], iters, (100, 20), u)
        end = time.time()
        elapsed += end - start
        pyplot.figure(figsize=(8.0, 8.0))
        pyplot.imshow(i, cmap="gray", vmin=0.0, vmax=1.0)
        pyplot.axis("off")
        pyplot.contour(u, [0.0], colors="red")
        pyplot.savefig("Figure01{}{:02}.png".format(filename, k+1), bbox_inches="tight")
        pyplot.show()
        pyplot.close()
        print("{} step {} finished, {:.3f}s elapsed".format(filename, k, end - start))
    rt[0][filename] = elapsed


# In[ ]:


with shelve.open("Result") as db:
    db[str((1, "filename"))] = filename_list
    db[str((1, "alpha"))] = alpha_list
    db[str((1, "tau"))] = tau_list
    db[str((1, "len"))] = len_list
    db[str((1, "edge"))] = edge_list
    db[str((1, "time"))] = rt[0]

