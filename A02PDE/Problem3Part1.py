
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import skimage.io
from utils import degrade
import shock


# In[2]:


filename_list = [
    "baboon", "barbara", "boats", "lena", "peppers",
    "gradient", "radial", "triangle",
    "konata", "kagami", "tsukasa", "miyuki"
]
nu_list = [
    1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0,
    1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0,
    1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0,
]
iter_list = [
    10, 10, 10, 10, 10,
    10, 20, 20,
    20, 20, 20, 20,
]


# In[3]:


sigma = 2.0


# In[4]:


rt = [[], [], [], [], [], []]


# In[6]:


for j, filename in enumerate(filename_list):
    nu, iter_ = nu_list[j], iter_list[j]
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, None)
    start = time.time()
    u, ctr = shock.evolve_shock(i_degr, nu, iter_)
    end = time.time()
    print("{} iterations".format(ctr))
    rt[0].append((filename, end - start))
    rt[1].append((filename, ctr))
    rt[2].append((filename, skimage.measure.compare_psnr(i, i_degr)))
    rt[3].append((filename, skimage.measure.compare_psnr(i, u)))
    rt[4].append((filename, skimage.measure.compare_ssim(i, i_degr)))
    rt[5].append((filename, skimage.measure.compare_ssim(i, u)))
    if j in [1, 3, 4, 5, 7, 10]:
        skimage.io.imsave("Figure07{}0.png".format(filename), numpy.clip(i, 0.0, 1.0))
        skimage.io.imsave("Figure07{}1.png".format(filename), numpy.clip(i_degr, 0.0, 1.0))
        skimage.io.imsave("Figure07{}2.png".format(filename), numpy.clip(u, 0.0, 1.0))
        print("{} saved".format(filename))
    print("{} finished".format(filename))


# In[7]:


with shelve.open("Result") as db:
    db[str((3, 1, "filename"))] = filename_list
    db[str((3, 1, "nu"))] = nu_list
    db[str((3, 1, "time"))] = rt[0]
    db[str((3, 1, "iter"))] = rt[1]
    db[str((3, 1, "origpsnr"))] = rt[2]
    db[str((3, 1, "psnr"))] = rt[3]
    db[str((3, 1, "origssim"))] = rt[4]
    db[str((3, 1, "ssim"))] = rt[5]

