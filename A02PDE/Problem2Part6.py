
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import skimage.io
from utils import degrade
import pm


# In[2]:


filename_list = [
    "baboon", "barbara", "boats", "lena", "peppers",
    "gradient", "radial", "triangle",
    "konata", "kagami", "tsukasa", "miyuki"
]
mu_list = [
    1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0,
    1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0,
    1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0,
]
iter_list = [
    500, 500, 500, 500, 500,
    1000, 500, 1000,
    500, 500, 500, 500
]


# In[3]:


sigma = 2.0
eta = 5.0 / 255.0


# In[4]:


rt = [[], [], [], [], [], []]


# In[5]:


for j, filename in enumerate(filename_list):
    mu, iter_ = mu_list[j], iter_list[j]
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, eta)
    start = time.time()
    u, ctr = pm.evolve_pm(i_degr, mu, iter_, 1.0, nonl="frac")
    end = time.time()
    print("{} iterations".format(ctr))
    rt[0].append((filename, end - start))
    rt[1].append((filename, ctr))
    rt[2].append((filename, skimage.measure.compare_psnr(i, i_degr)))
    rt[3].append((filename, skimage.measure.compare_psnr(i, u)))
    rt[4].append((filename, skimage.measure.compare_ssim(i, i_degr)))
    rt[5].append((filename, skimage.measure.compare_ssim(i, u)))
    if j in [1, 3, 5, 10]:
        skimage.io.imsave("Figure11{}0.png".format(filename), numpy.clip(i, 0.0, 1.0))
        skimage.io.imsave("Figure11{}1.png".format(filename), numpy.clip(i_degr, 0.0, 1.0))
        skimage.io.imsave("Figure11{}2.png".format(filename), numpy.clip(u, 0.0, 1.0))
        print("{} saved".format(filename))
    print("{} finished".format(filename))


# In[6]:


with shelve.open("Result") as db:
    db[str((2, 6, "filename"))] = filename_list
    db[str((2, 6, "mu"))] = mu_list
    db[str((2, 6, "time"))] = rt[0]
    db[str((2, 6, "iter"))] = rt[1]
    db[str((2, 6, "origpsnr"))] = rt[2]
    db[str((2, 6, "psnr"))] = rt[3]
    db[str((2, 6, "origssim"))] = rt[4]
    db[str((2, 6, "ssim"))] = rt[5]

