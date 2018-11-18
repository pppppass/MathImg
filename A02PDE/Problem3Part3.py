
# coding: utf-8

# In[11]:


import numpy
import skimage.io
from utils import degrade
import shock


# In[12]:


filename_list = ["lena", "tsukasa"]
nu_list = [1.0 / 40.0, 1.0 / 40.0]
iter_list = [80, 80]


# In[13]:


sigma = 2.0
eta = 5.0 / 255.0


# In[14]:


for j, filename in enumerate(filename_list):
    nu = nu_list[j]
    iter_ = iter_list[j]
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, None)
    u, _ = shock.evolve_shock(i_degr, nu, iter_)
    skimage.io.imsave("Figure09{}1.png".format(filename), numpy.clip(u, 0.0, 1.0))
    i_degr = degrade.calc_degrade(i, sigma, eta)
    u, _ = shock.evolve_shock(i_degr, nu, iter_)
    skimage.io.imsave("Figure09{}2.png".format(filename), numpy.clip(u, 0.0, 1.0))
    print("{} finished".format(filename))

