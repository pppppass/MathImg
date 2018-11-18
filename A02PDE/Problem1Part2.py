
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import skimage.io
from utils import degrade
import heat


# In[2]:


filename_list = ["barbara", "lena", "tsukasa"]
mu_list = [1.0 / 200.0, 1.0 / 200.0, 1.0 / 200.0]
iter_list = [
    [15, 30, 60, 120, 240],
    [30, 60, 120, 240, 480],
    [25, 50, 100, 200, 400],
]


# In[3]:


eta = 20.0 / 255.0


# In[4]:


rt = [[], []]


# In[5]:


for j, filename in enumerate(filename_list):
    mu = mu_list[j]
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, None, eta)
    rt[0].append((filename, -1, skimage.measure.compare_psnr(i, i_degr)))
    rt[1].append((filename, -1, skimage.measure.compare_ssim(i, i_degr)))
    skimage.io.imsave("Figure02{}0.png".format(filename), numpy.clip(i_degr, 0.0, 1.0))
    for k, iter_ in enumerate(iter_list[j]):
        u, _ = heat.evolve_heat(i_degr, mu, iter_)
        rt[0].append((filename, iter_, skimage.measure.compare_psnr(i, u)))
        rt[1].append((filename, iter_, skimage.measure.compare_ssim(i, u)))
        skimage.io.imsave("Figure02{}{}.png".format(filename, k+1), numpy.clip(u, 0.0, 1.0))
        print("iter = {} finished".format(iter_))
    print("{} finished".format(filename))


# In[6]:


with shelve.open("Result") as db:
    db[str((1, 3, "filename"))] = filename_list
    db[str((1, 3, "mu"))] = mu_list
    db[str((1, 3, "iter"))] = iter_list
    db[str((1, 3, "psnr"))] = rt[0]
    db[str((1, 3, "ssim"))] = rt[1]

