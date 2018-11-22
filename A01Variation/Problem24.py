
# coding: utf-8

# In[1]:


import numpy
import skimage.io
from utils import degrade
import tv


# In[2]:


filenames = ["lena", "tsukasa"]
lambdas = [4.0e-6, 4.0e-6]
rhos = [2.0e-6, 2.0e-6]


# In[3]:


sigma = 2.0
eta = 5.0 / 255.0


# In[4]:


for j in range(len(filenames)):
    i = skimage.io.imread("dataset/{}.bmp".format(filenames[j])) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, eta)
    u, _ = tv.opt_tv_admm(i_degr, sigma, lambdas[j], rhos[j], 1000, 1.0e-3, tv="aniso")
    skimage.io.imsave("Figure4{}1.png".format(filenames[j]), numpy.clip(u, 0.0, 1.0))
    print("{} anisotropic finished".format(filenames[j]))
    u, _ = tv.opt_tv_admm(i_degr, sigma, lambdas[j], rhos[j], 1000, 1.0e-3, tv="iso")
    skimage.io.imsave("Figure4{}2.png".format(filenames[j]), numpy.clip(u, 0.0, 1.0))
    print("{} isotropic finished".format(filenames[j]))

