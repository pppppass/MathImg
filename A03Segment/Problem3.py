
# coding: utf-8

# In[1]:


import numpy
import skimage.io
from matplotlib import pyplot
from utils import degrade, downsamp
import gac


# In[2]:


filename_list = [
    "triangle",
    "bird",
]
alpha_list = [
    150.0,
    100.0,
]
tau_list = [
    0.012 * (1.0 / 256)**2,
    0.02 * (1.0 / 160)**2,
]
edge_list = [
    0.3,
    0.3,
]
len_list = [
    (2000, 10),
    (2250, 10),
]


# In[3]:


for j, filename in enumerate(filename_list):
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i = downsamp.samp_down(i)
    iters, len_ = len_list[j]
    i_degr = degrade.calc_degrade(i, 2.0, None)
    u = gac.calc_init(i, 0.02)
    for k in range(len_):
        u = gac.evolve_gac(i_degr, alpha_list[j], tau_list[j], edge_list[j], iters, (100, 20), u)
        pyplot.figure(figsize=(8.0, 8.0))
        pyplot.imshow(i_degr, cmap="gray", vmin=0.0, vmax=1.0)
        pyplot.axis("off")
        pyplot.contour(u, [0.0], colors="red")
        pyplot.savefig("Figure03{}{:02}.png".format(filename, k+1), bbox_inches="tight")
        pyplot.show()
        pyplot.close()
        print("{} step {} finished".format(filename, k))
    i_degr = degrade.calc_degrade(i, None, 30.0 / 255.0)
    u = gac.calc_init(i, 0.02)
    for k in range(len_):
        u = gac.evolve_gac(i_degr, alpha_list[j], tau_list[j], edge_list[j], iters, (100, 20), u)
        pyplot.figure(figsize=(8.0, 8.0))
        pyplot.imshow(i_degr, cmap="gray", vmin=0.0, vmax=1.0)
        pyplot.axis("off")
        pyplot.contour(u, [0.0], colors="red")
        pyplot.savefig("Figure04{}{:02}.png".format(filename, k+1), bbox_inches="tight")
        pyplot.show()
        pyplot.close()
        print("{} step {} finished".format(filename, k))

