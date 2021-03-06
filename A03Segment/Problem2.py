
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import skimage.io
from matplotlib import pyplot
import cv


# In[2]:


filename_list = [
    "triangle",
    "objects",
    "cells",
    "bird",
    "lena",
    "konata",
]
mu_list = [
    2000.0,
    2000.0,
    20000.0,
    2000.0,
    2000.0,
    2000.0,
]
c_list = [
    (1.0, 0.0),
    (1.0, 0.0),
    (0.3, 0.0),
    (1.0, 0.0),
    (1.0, 0.0),
    (1.0, 0.3),
]


# In[3]:


rt = [{}, {}]


# In[4]:


for j, filename in enumerate(filename_list):
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    start = time.time()
    u, c1, c2 = cv.evolve_cv(i, mu_list[j], *c_list[j], 0.001, 0.001, 30, 30)
    end = time.time()
    pyplot.figure(figsize=(8.0, 8.0))
    pyplot.imshow(i, cmap="gray", vmin=0.0, vmax=1.0)
    pyplot.axis("off")
    pyplot.contour(u, [0.5], colors="red")
    pyplot.savefig("Figure02{}.png".format(filename), bbox_inches="tight")
    pyplot.show()
    pyplot.close()
    print("{} finished".format(filename))
    rt[0][filename] = end - start
    rt[1][filename] = (c1, c2)


# In[5]:


with shelve.open("Result") as db:
    db[str((2, "filename"))] = filename_list
    db[str((2, "mu"))] = mu_list
    db[str((2, "c", "init"))] = c_list
    db[str((2, "c", "final"))] = rt[1]
    db[str((2, "time"))] = rt[0]

