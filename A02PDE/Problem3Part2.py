
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import skimage.io
from utils import degrade
import shock


# In[2]:


filename_list = ["lena", "tsukasa"]
nu_list = [1.0 / 40.0, 1.0 / 40.0]
iter_list = [
    [10, 20, 40, 80, 160],
    [10, 20, 40, 80, 160],
]


# In[3]:


sigma = 2.0


# In[4]:


rt = [[], []]


# In[5]:


for j, filename in enumerate(filename_list):
    nu = nu_list[j]
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, None)
    rt[0].append((filename, -1, skimage.measure.compare_psnr(i, i_degr)))
    rt[1].append((filename, -1, skimage.measure.compare_ssim(i, i_degr)))
    skimage.io.imsave("Figure08{}0.png".format(filename), numpy.clip(i_degr, 0.0, 1.0))
    for k, iter_ in enumerate(iter_list[j]):
        u, _ = shock.evolve_shock(i_degr, nu, iter_)
        rt[0].append((filename, iter_, skimage.measure.compare_psnr(i, u)))
        rt[1].append((filename, iter_, skimage.measure.compare_ssim(i, u)))
        skimage.io.imsave("Figure08{}{}.png".format(filename, k+1), numpy.clip(u, 0.0, 1.0))
        print("iter = {} finished".format(iter_))
    print("{} finished".format(filename))


# In[6]:


with shelve.open("Result") as db:
    db[str((3, 2, "filename"))] = filename_list
    db[str((3, 2, "nu"))] = nu_list
    db[str((3, 2, "iter"))] = iter_list
    db[str((3, 2, "psnr"))] = rt[0]
    db[str((3, 2, "ssim"))] = rt[1]

