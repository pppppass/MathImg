
# coding: utf-8

# In[3]:


import os.path
import shelve
import numpy
import skimage.io
from utils import degrade
import analysis


# In[2]:


filename_list = [
    "lena", "tsukasa"
]
lambda_list = [
    [0.00075, 0.0015, 0.003, 0.006, 0.012],
    [0.00075, 0.0015, 0.003, 0.006, 0.012]
]
rho_list = [
    1.0, 1.0
]
wavelet_name_list = [
    "db6", "haar"
]


# In[3]:


sigma = 2.0
eta = 5.0 / 255.0


# In[4]:


desk = [{}, {}]
if os.path.exists("Problem1Part3.run"):
    with shelve.open("Result") as db:
        desk[0] = db[str((1, 3, "psnr"))]
        desk[1] = db[str((1, 3, "ssim"))]


# In[5]:


for j, filename in enumerate(filename_list):
    desk[0][filename] = []
    desk[1][filename] = []
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, eta)
    desk[0][filename].append(skimage.measure.compare_psnr(i, i_degr))
    desk[1][filename].append(skimage.measure.compare_ssim(i, i_degr))
    skimage.io.imsave("Figure4{}0.png".format(filename), numpy.clip(i_degr, 0.0, 1.0))
    for k, lamda in enumerate(lambda_list[j]):
        u, ctr = analysis.opt_analysis_admm(i_degr, sigma, lamda, wavelet_name_list[j], 4, rho_list[j], 1000, 5.0e-5)
        desk[0][filename].append(skimage.measure.compare_psnr(i, u))
        desk[1][filename].append(skimage.measure.compare_ssim(i, u))
        skimage.io.imsave("Figure4{}{}.png".format(filename, k+1), numpy.clip(u, 0.0, 1.0))
        print("lambda = {} finished".format(lamda))
    print("{} finished".format(filename))


# In[ ]:


with shelve.open("Result") as db:
    db[str((1, 3, "filename"))] = filename_list
    db[str((1, 3, "lambda"))] = lambda_list
    db[str((1, 3, "rho"))] = rho_list
    db[str((1, 3, "wavelet", "name"))] = wavelet_name_list
    db[str((1, 3, "psnr"))] = desk[0]
    db[str((1, 3, "ssim"))] = desk[1]

