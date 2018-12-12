
# coding: utf-8

# In[1]:


import numpy
import skimage.io
from matplotlib import pyplot
from utils import degrade
import cv


# In[2]:


filename_list = [
    "triangle",
    "bird",
]
mu_list = [
    2000.0,
    2000.0,
]
c_list = [
    (1.0, 0.0),
    (1.0, 0.0),
]


# In[3]:


for j, filename in enumerate(filename_list):
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, 4.0, None)
    u, c1, c2 = cv.evolve_cv(i_degr, mu_list[j], *c_list[j], 0.001, 0.001, 30, 30)
    pyplot.figure(figsize=(8.0, 8.0))
    pyplot.imshow(i_degr, cmap="gray", vmin=0.0, vmax=1.0)
    pyplot.axis("off")
    pyplot.contour(u, [0.5], colors="red")
    pyplot.savefig("Figure05{}.png".format(filename), bbox_inches="tight")
    pyplot.show()
    pyplot.close()
    i_degr = degrade.calc_degrade(i, None, 30.0 / 255.0)
    u, c1, c2 = cv.evolve_cv(i_degr, mu_list[j], *c_list[j], 0.001, 0.001, 30, 30)
    pyplot.figure(figsize=(8.0, 8.0))
    pyplot.imshow(i_degr, cmap="gray", vmin=0.0, vmax=1.0)
    pyplot.axis("off")
    pyplot.contour(u, [0.5], colors="red")
    pyplot.savefig("Figure06{}.png".format(filename), bbox_inches="tight")
    pyplot.show()
    pyplot.close()
    print("{} finished".format(filename))

