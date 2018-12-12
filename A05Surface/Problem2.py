
# coding: utf-8

# In[1]:


import shelve
import numpy
import skimage.io
from matplotlib import pyplot
import cv
import wcv


# In[2]:


mu_list = [
    [500.0, 1000.0, 2000.0, 4000.0, 8000.0],
    [1500.0, 3000.0, 6000.0, 12000.0, 24000.0],
    [1500.0, 3000.0, 6000.0, 12000.0, 24000.0],
    [1500.0, 3000.0, 6000.0, 12000.0, 24000.0],
]
wavelet_name_list = [
    "haar",
    "db6",
    "coif2",
]
wavelet_filename_list = [
    "tv",
    "haar",
    "db",
    "coif"
]


# In[3]:


i = skimage.io.imread("dataset/bird.bmp") / 255.0
pyplot.figure(figsize=(8.0, 8.0))
pyplot.imshow(i, cmap="gray", vmin=0.0, vmax=1.0)
pyplot.axis("off")
pyplot.savefig("Figure30.png", bbox_inches="tight")
pyplot.show()
pyplot.close()
for j, mu in enumerate(mu_list[0]):
    u, _, _ = cv.evolve_cv(i, mu, 1.0, 0.0, 0.001, 0.001, 30, 30)
    pyplot.figure(figsize=(8.0, 8.0))
    pyplot.imshow(i, cmap="gray", vmin=0.0, vmax=1.0)
    pyplot.axis("off")
    pyplot.contour(u, [0.5], colors="red")
    pyplot.savefig("Figure3{}{}.png".format(wavelet_filename_list[0], j+1), bbox_inches="tight")
    pyplot.show()
    pyplot.close()
    print("tv, mu = {} finished".format(mu))
for k, wavelet_name in enumerate(wavelet_name_list):
    for j, mu in enumerate(mu_list[k+1]):
        u, c1, c2 = wcv.evolve_wcv(i, mu, wavelet_name, 2, 1.0, 0.0, 0.001, 0.001, 30, 30)
        pyplot.figure(figsize=(8.0, 8.0))
        pyplot.imshow(i, cmap="gray", vmin=0.0, vmax=1.0)
        pyplot.axis("off")
        pyplot.contour(u, [0.5], colors="red")
        pyplot.savefig("Figure03{}{}.png".format(wavelet_filename_list[k+1], j+1), bbox_inches="tight")
        pyplot.show()
        pyplot.close()
        print("{}, mu = {} finished".format(wavelet_name, mu))


# In[4]:


with shelve.open("Result") as db:
    db[str((2, "wavelet", "name"))] = wavelet_name_list
    db[str((2, "mu"))] = mu_list
    db[str((2, "wavelet", "filename"))] = wavelet_filename_list

