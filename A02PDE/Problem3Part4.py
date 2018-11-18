
# coding: utf-8

# In[4]:


import numpy
import skimage.io
from utils import degrade
import shock


# In[5]:


filename_list = ["lena", "tsukasa"]
nu_list = [1.0 / 40.0, 1.0 / 40.0]
iter_list = [80, 80]


# In[6]:


sigma = 2.0


# In[7]:


for j, filename in enumerate(filename_list):
    nu = nu_list[j]
    iter_ = iter_list[j]
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, sigma, None)
    u, _ = shock.evolve_shock(i_degr, nu, iter_, ind="lap")
    skimage.io.imsave("Figure10{}1.png".format(filename), numpy.clip(u, 0.0, 1.0))
    u, _ = shock.evolve_shock(i_degr, nu, iter_, ind="curve")
    skimage.io.imsave("Figure10{}2.png".format(filename), numpy.clip(u, 0.0, 1.0))
    print("{} finished".format(filename))

