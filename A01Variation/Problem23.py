
# coding: utf-8

# In[1]:


import numpy
import skimage.io
from utils import degrade
import tv


# In[2]:


filenames = ["lena", "tsukasa"]
lambdas = [2.0e-6, 2.0e-6]
rhos = [2.0e-6, 2.0e-6]


# In[3]:


sigma = 2.0
eta = 5.0 / 255.0


# In[4]:


for j in range(len(filenames)):
    i = skimage.io.imread("dataset/{}.bmp".format(filenames[j])) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, eta)
    u, _ = tv.opt_tv_admm(i_degr, sigma, lambdas[j], rhos[j], 1000, 1.0e-3, inv="dct")
    skimage.io.imsave("Figure3{}1.png".format(filenames[j]), numpy.clip(u, 0.0, 1.0))
    print("{} DCT finished".format(filenames[j]))
    u, _ = tv.opt_tv_admm(i_degr, sigma, lambdas[j], rhos[j], 1000, 1.0e-3, inv="fft")
    skimage.io.imsave("Figure3{}2.png".format(filenames[j]), numpy.clip(u, 0.0, 1.0))
    print("{} FFT finished".format(filenames[j]))
    u, _ = tv.opt_tv_admm(i_degr, sigma, lambdas[j], rhos[j], 1000, 1.0e-3, inv="dst")
    skimage.io.imsave("Figure3{}3.png".format(filenames[j]), numpy.clip(u, 0.0, 1.0))
    print("{} DST finished".format(filenames[j]))

