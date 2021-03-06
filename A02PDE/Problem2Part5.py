
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import skimage.io
from utils import degrade
import pm


# In[2]:


filename_list = ["barbara", "lena", "tsukasa"]
mu_list = [1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0]
iter_list = [
    [32, 64, 128, 256, 512],
    [18, 36, 72, 144, 288],
    [16, 32, 64, 128, 256]
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
    skimage.io.imsave("Figure06{}0.png".format(filename), numpy.clip(i_degr, 0.0, 1.0))
    for k, iter_ in enumerate(iter_list[j]):
        u, _ = pm.evolve_pm(i_degr, mu, iter_, 100.0, nonl="frac")
        rt[0].append((filename, iter_, skimage.measure.compare_psnr(i, u)))
        rt[1].append((filename, iter_, skimage.measure.compare_ssim(i, u)))
        skimage.io.imsave("Figure06{}{}.png".format(filename, k+1), numpy.clip(u, 0.0, 1.0))
        print("iter = {} finished".format(iter_))
    print("{} finished".format(filename))


# In[6]:


with shelve.open("Result") as db:
    db[str((2, 5, "filename"))] = filename_list
    db[str((2, 5, "mu"))] = mu_list
    db[str((2, 5, "iter"))] = iter_list
    db[str((2, 5, "psnr"))] = rt[0]
    db[str((2, 5, "ssim"))] = rt[1]

