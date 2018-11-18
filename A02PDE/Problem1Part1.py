
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import skimage.io
from utils import degrade
import heat


# In[2]:


filename_list = [
    "baboon", "barbara", "boats", "lena", "peppers",
    "gradient", "radial", "triangle",
    "konata", "kagami", "tsukasa", "miyuki"
]
mu_list = [
    1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0,
    1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0,
    1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0,
]
iter_list = [
    3, 3, 5, 6, 6,
    10, 3, 3,
    5, 5, 5, 5
]


# In[3]:


eta = 20.0 / 255.0


# In[4]:


rt = [[], [], [], [], [], []]


# In[5]:


for j, filename in enumerate(filename_list):
    mu, iter_ = mu_list[j], iter_list[j]
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, None, eta)
    start = time.time()
    u, ctr = heat.evolve_heat(i_degr, mu, iter_)
    end = time.time()
    print("{} iterations".format(ctr))
    rt[0].append((filename, end - start))
    rt[1].append((filename, ctr))
    rt[2].append((filename, skimage.measure.compare_psnr(i, i_degr)))
    rt[3].append((filename, skimage.measure.compare_psnr(i, u)))
    rt[4].append((filename, skimage.measure.compare_ssim(i, i_degr)))
    rt[5].append((filename, skimage.measure.compare_ssim(i, u)))
    if j in [0, 1, 3, 5, 6, 10]:
        skimage.io.imsave("Figure01{}0.png".format(filename), numpy.clip(i, 0.0, 1.0))
        skimage.io.imsave("Figure01{}1.png".format(filename), numpy.clip(i_degr, 0.0, 1.0))
        skimage.io.imsave("Figure01{}2.png".format(filename), numpy.clip(u, 0.0, 1.0))
        print("{} saved".format(filename))
    print("{} finished".format(filename))


# In[6]:


with shelve.open("Result") as db:
    db[str((1, 2, "filename"))] = filename_list
    db[str((1, 2, "mu"))] = mu_list
    db[str((1, 2, "time"))] = rt[0]
    db[str((1, 2, "iter"))] = rt[1]
    db[str((1, 2, "origpsnr"))] = rt[2]
    db[str((1, 2, "psnr"))] = rt[3]
    db[str((1, 2, "origssim"))] = rt[4]
    db[str((1, 2, "ssim"))] = rt[5]

