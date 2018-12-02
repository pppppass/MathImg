
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
    0.003, 0.003
]
rho_list = [
    1.0, 1.0
]
wavelet_name_list = [
    "haar", "db2", "db6", "sym8", "coif2", "bior2.6",
]


# In[3]:


sigma = 2.0
eta = 5.0 / 255.0


# In[4]:


desk = [{}, {}]
if os.path.exists("Problem1Part4.run"):
    with shelve.open("Result") as db:
        desk[0] = db[str((1, 4, "psnr"))]
        desk[1] = db[str((1, 4, "ssim"))]


# In[5]:


for j, filename in enumerate(filename_list):
    desk[0][filename] = []
    desk[1][filename] = []
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, eta)
    for k, wavelet_name in enumerate(wavelet_name_list):
        u, ctr = analysis.opt_analysis_admm(i_degr, sigma, lambda_list[j], wavelet_name, 4, rho_list[j], 1000, 5.0e-5)
        desk[0][filename].append(skimage.measure.compare_psnr(i, u))
        desk[1][filename].append(skimage.measure.compare_ssim(i, u))
        skimage.io.imsave("Figure5{}{}.png".format(filename, k+1), numpy.clip(u, 0.0, 1.0))
        print("Wavelet {} finished".format(wavelet_name))
    print("{} finished".format(filename))


# In[ ]:


with shelve.open("Result") as db:
    db[str((1, 4, "filename"))] = filename_list
    db[str((1, 4, "lambda"))] = lambda_list
    db[str((1, 4, "rho"))] = rho_list
    db[str((1, 4, "wavelet", "name"))] = wavelet_name_list
    db[str((1, 4, "psnr"))] = desk[0]
    db[str((1, 4, "ssim"))] = desk[1]

