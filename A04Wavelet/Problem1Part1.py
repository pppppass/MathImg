
# coding: utf-8

# In[3]:


import os.path
import time
import shelve
import numpy
import skimage.io
from utils import degrade
import analysis


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
rho_list = [
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0
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
if os.path.exists("Problem1Part1.run"):
    with shelve.open("Result") as db:
        desk[0] = db[str((1, 1, "time"))]
        desk[1] = db[str((1, 1, "iter"))]
        desk[2] = db[str((1, 1, "psnr", "before"))]
        desk[3] = db[str((1, 1, "psnr", "after"))]
        desk[4] = db[str((1, 1, "ssim", "before"))]
        desk[5] = db[str((1, 1, "ssim", "after"))]


# In[5]:


for j, filename in enumerate(filename_list):
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, eta)
    start = time.time()
    u, ctr = analysis.opt_analysis_admm(i_degr, sigma, lambda_list[j], wavelet_name_list[j], 4, rho_list[j], 1000, 5.0e-5)
    end = time.time()
    desk[0][filename] = end - start
    desk[1][filename] = ctr
    desk[2][filename] = skimage.measure.compare_psnr(i, i_degr)
    desk[3][filename] = skimage.measure.compare_psnr(i, u)
    desk[4][filename] = skimage.measure.compare_ssim(i, i_degr)
    desk[5][filename] = skimage.measure.compare_ssim(i, u)
    if j in [1, 2, 3, 5, 7, 10]:
        skimage.io.imsave("Figure1{}0.png".format(filename), numpy.clip(i, 0.0, 1.0))
        skimage.io.imsave("Figure1{}1.png".format(filename), numpy.clip(i_degr, 0.0, 1.0))
        skimage.io.imsave("Figure1{}2.png".format(filename), numpy.clip(u, 0.0, 1.0))
        print("{} saved".format(filename))
    print("{} finished, PSNR = {:.5f}, SSIM = {:.5f}".format(filename, desk[3][filename], desk[5][filename]))


# In[ ]:


with shelve.open("Result") as db:
    db[str((1, 1, "filename"))] = filename_list
    db[str((1, 1, "lambda"))] = lambda_list
    db[str((1, 1, "rho"))] = rho_list
    db[str((1, 1, "wavelet", "name"))] = wavelet_name_list
    db[str((1, 1, "time"))] = desk[0]
    db[str((1, 1, "iter"))] = desk[1]
    db[str((1, 1, "psnr", "before"))] = desk[2]
    db[str((1, 1, "psnr", "after"))] = desk[3]
    db[str((1, 1, "ssim", "before"))] = desk[4]
    db[str((1, 1, "ssim", "after"))] = desk[5]

