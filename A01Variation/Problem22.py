
# coding: utf-8

# In[1]:


import shelve
import numpy
import skimage
import tv


# In[2]:


sigma = 2.0
eta = 5.0 / 255.0


# In[3]:


filenames = ["lena", "tsukasa"]
lambdas = [
    [0.5e-6, 1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6],
    [0.5e-6, 1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6]
]
rhos = [
    [0.5e-6, 1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6],
    [0.5e-6, 1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6]
]


# In[4]:


rt = [[[], []], [[], []]]


# In[5]:


for j in range(len(filenames)):
    i = skimage.io.imread("dataset/{}.bmp".format(filenames[j])) / 255.0
    i_degr = tv.calc_degrade(i, sigma, eta)
    rt[j].append(skimage.measure.compare_psnr(i, i_degr))
    skimage.io.imsave("Figure2{}0.bmp".format(filenames[j]), numpy.clip(i_degr, 0.0, 1.0))
    for k in range(len(lambdas[j])):
        u, _ = tv.opt_tv_admm(i_degr, sigma, lambdas[j][k], rhos[j][k], 1000, 1.0e-3)
        rt[j][0].append(skimage.measure.compare_psnr(i, u))
        rt[j][1].append(skimage.measure.compare_ssim(i, u))
        skimage.io.imsave("Figure2{}{}.bmp".format(filenames[j], k+1), numpy.clip(u, 0.0, 1.0))
        print("k = {} finished".format(k))
    print("{} finished".format(filenames[j]))


# In[6]:


with shelve.open("Result") as db:
    db["2filename"] = filenames
    db["2lambda"] = lambdas
    db["2rho"] = rhos
    db["2result"] = rt

