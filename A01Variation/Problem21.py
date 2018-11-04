
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import skimage
import tv


# In[2]:


filenames = [
    "baboon", "barbara", "boats", "lena", "peppers",
    "gradient", "radial", "triangle",
    "konata", "kagami", "tsukasa", "miyuki"
]
lambdas = [
    2.0e-6, 2.5e-6, 2.0e-6, 2.0e-6, 4.0e-6,
    5.0e-6, 5.0e-6, 5.0e-6,
    2.0e-6, 2.0e-6, 2.0e-6, 2.0e-6, 
]
rhos = [
    2.0e-6, 2.5e-6, 2.0e-6, 2.0e-6, 4.0e-6,
    1.0e-6, 1.0e-6, 1.0e-6,
    2.0e-6, 2.0e-6, 2.0e-6, 2.0e-6,
]


# In[3]:


sigma = 2.0
eta = 5.0 / 255.0


# In[4]:


rt = [[], [], [], [], [], []]


# In[5]:


for j in range(len(filenames)):
    i = skimage.io.imread("dataset/{}.bmp".format(filenames[j])) / 255.0
    i_degr = tv.calc_degrade(i, sigma, eta)
    start = time.time()
    u, ctr = tv.opt_tv_admm(i_degr, sigma, lambdas[j], rhos[j], 1000, 1.0e-3)
    end = time.time()
    rt[0].append(end - start)
    rt[1].append(ctr)
    rt[2].append(skimage.measure.compare_psnr(i, i_degr))
    rt[3].append(skimage.measure.compare_psnr(i, u))
    rt[4].append(skimage.measure.compare_ssim(i, i_degr))
    rt[5].append(skimage.measure.compare_ssim(i, u))
    if j in [1, 2, 3, 5, 7, 10]:
        skimage.io.imsave("Figure1{}0.png".format(filenames[j]), numpy.clip(i, 0.0, 1.0))
        skimage.io.imsave("Figure1{}1.png".format(filenames[j]), numpy.clip(i_degr, 0.0, 1.0))
        skimage.io.imsave("Figure1{}2.png".format(filenames[j]), numpy.clip(u, 0.0, 1.0))
        print("{} saved".format(filenames[j]))
    print("{} finished".format(filenames[j]))


# In[6]:


with shelve.open("Result") as db:
    db["1filename"] = filenames
    db["1lambda"] = lambdas
    db["1rho"] = rhos
    db["1result"] = rt

