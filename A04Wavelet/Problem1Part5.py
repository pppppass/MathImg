
# coding: utf-8

# In[4]:


import numpy
import skimage.io
from utils import degrade
import analysis


# In[5]:


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
    "db6", "haar"
]


# In[6]:


sigma = 2.0
eta = 5.0 / 255.0


# In[7]:


for j, filename in enumerate(filename_list):
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, eta)
    skimage.io.imsave("Figure8{}0.png".format(filename), numpy.clip(i, 0.0, 1.0))
    u, ctr = analysis.opt_analysis_admm(i_degr, sigma, lambda_list[j], wavelet_name_list[j], 4, rho_list[j], 1000, 5.0e-5)
    skimage.io.imsave("Figure8{}1.png".format(filename), numpy.clip(u, 0.0, 1.0))
    u, ctr = analysis.opt_analysis_admm(i_degr, sigma, lambda_list[j], wavelet_name_list[j], 4, rho_list[j], 1000, 5.0e-5, wavelet_style="dwt")
    skimage.io.imsave("Figure8{}2.png".format(filename), numpy.clip(u, 0.0, 1.0))
    print("{} finished".format(filename))

