
# coding: utf-8

# In[1]:


import os.path
import shelve
import numpy
import skimage.io
from utils import degrade
import balance


# In[2]:


filename_list = [
    "lena", "tsukasa"
]
lambda_list = [
    [0.00075, 0.0015, 0.003, 0.006, 0.012],
    [0.00075, 0.0015, 0.003, 0.006, 0.012]
]
wavelet_name_list = [
    "db6", "haar"
]


# In[3]:


sigma = 2.0
eta = 5.0 / 255.0


# In[4]:


desk = [{}, {}]
if os.path.exists("Problem2Part2.run"):
    with shelve.open("Result") as db:
        desk[0] = db[str((2, 2, "psnr"))]
        desk[1] = db[str((2, 2, "ssim"))]


# In[6]:


for j, filename in enumerate(filename_list):
    desk[0][filename] = []
    desk[1][filename] = []
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, eta)
    desk[0][filename].append(skimage.measure.compare_psnr(i, i_degr))
    desk[1][filename].append(skimage.measure.compare_ssim(i, i_degr))
    skimage.io.imsave("Figure6{}0.png".format(filename), numpy.clip(i, 0.0, 1.0))
    for k, lamda in enumerate(lambda_list[j]):
        u, ctr = balance.opt_balance_prox(i_degr, sigma, lamda, 0.3, wavelet_name_list[j], 4, 0.1, 1000, 5.0e-4, nesterov=True)
        desk[0][filename].append(skimage.measure.compare_psnr(i, u))
        desk[1][filename].append(skimage.measure.compare_ssim(i, u))
        skimage.io.imsave("Figure6{}{}.png".format(filename, k+1), numpy.clip(u, 0.0, 1.0))
        print("lambda = {} finished".format(lamda))
    print("{} finished".format(filename))


# In[ ]:


with shelve.open("Result") as db:
    db[str((2, 2, "filename"))] = filename_list
    db[str((2, 2, "lambda"))] = lambda_list
    db[str((2, 2, "wavelet", "name"))] = wavelet_name_list
    db[str((2, 2, "psnr"))] = desk[0]
    db[str((2, 2, "ssim"))] = desk[1]

