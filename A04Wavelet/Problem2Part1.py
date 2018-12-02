
# coding: utf-8

# In[1]:


import os.path
import time
import shelve
import numpy
import skimage.io
from utils import degrade
import balance


# In[2]:


filename_list = [
    "baboon", "barbara", "boats", "lena", "peppers",
    "gradient", "radial", "triangle",
    "konata", "kagami", "tsukasa", "miyuki"
]
lambda_list = [
    0.002, 0.002, 0.002, 0.003, 0.003,
    0.01, 0.01, 0.01,
    0.003, 0.003, 0.003, 0.003
]
wavelet_name_list = [
    "db6", "db6", "db6", "db6", "db6",
    "haar", "haar", "haar",
    "haar", "haar", "haar", "haar"
]


# In[3]:


sigma = 2.0
eta = 5.0 / 255.0


# In[4]:


desk = [{}, {}, {}, {}, {}, {}]
if os.path.exists("Problem2Part1.run"):
    with shelve.open("Result") as db:
        desk[0] = db[str((2, 1, "time"))]
        desk[1] = db[str((2, 1, "iter"))]
        desk[2] = db[str((2, 1, "psnr", "before"))]
        desk[3] = db[str((2, 1, "psnr", "after"))]
        desk[4] = db[str((2, 1, "ssim", "before"))]
        desk[5] = db[str((2, 1, "ssim", "after"))]


# In[6]:


for j, filename in enumerate(filename_list):
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, eta)
    start = time.time()
    u, ctr = balance.opt_balance_prox(i_degr, sigma, lambda_list[j], 0.3, wavelet_name_list[j], 4, 0.1, 1000, 5.0e-4, nesterov=True)
    end = time.time()
    desk[0][filename] = end - start
    desk[1][filename] = ctr
    desk[2][filename] = skimage.measure.compare_psnr(i, i_degr)
    desk[3][filename] = skimage.measure.compare_psnr(i, u)
    desk[4][filename] = skimage.measure.compare_ssim(i, i_degr)
    desk[5][filename] = skimage.measure.compare_ssim(i, u)
    if j in [1, 2, 3, 5, 7, 10]:
        skimage.io.imsave("Figure2{}0.png".format(filename), numpy.clip(i, 0.0, 1.0))
        skimage.io.imsave("Figure2{}1.png".format(filename), numpy.clip(i_degr, 0.0, 1.0))
        skimage.io.imsave("Figure2{}2.png".format(filename), numpy.clip(u, 0.0, 1.0))
        print("{} saved".format(filename))
    print("{} finished, PSNR = {:.5f}, SSIM = {:.5f}".format(filename, desk[3][filename], desk[5][filename]))


# In[ ]:


with shelve.open("Result") as db:
    db[str((2, 1, "filename"))] = filename_list
    db[str((2, 1, "lambda"))] = lambda_list
    db[str((2, 1, "wavelet", "name"))] = wavelet_name_list
    db[str((2, 1, "time"))] = desk[0]
    db[str((2, 1, "iter"))] = desk[1]
    db[str((2, 1, "psnr", "before"))] = desk[2]
    db[str((2, 1, "psnr", "after"))] = desk[3]
    db[str((2, 1, "ssim", "before"))] = desk[4]
    db[str((2, 1, "ssim", "after"))] = desk[5]

