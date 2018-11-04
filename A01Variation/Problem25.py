
# coding: utf-8

# In[1]:


import shelve
import numpy
import skimage
import tv


# In[2]:


sigma = 2.0
eta = 5.0 / 255.0


# In[7]:


filenames = ["lena", "tsukasa"]
lambdas = [
    [2.0e-4, 1.0e-4, 5.0e-5, 2.5e-5, 1.25e-5],
    [2.0e-4, 1.0e-4, 5.0e-5, 2.5e-5, 1.25e-5]
]
rhos = [
    [2.0e-6, 2.0e-6, 2.0e-6, 2.0e-6, 2.0e-6],
    [2.0e-6, 2.0e-6, 2.0e-6, 2.0e-6, 2.0e-6]
]


# In[8]:


rt = [[[], []], [[], []]]


# In[9]:


for j in range(len(filenames)):
    i = skimage.io.imread("dataset/{}.bmp".format(filenames[j])) / 255.0
    i_degr = tv.calc_degrade(i, sigma, eta)
    rt[j].append(skimage.measure.compare_psnr(i, i_degr))
    skimage.io.imsave("Figure2{}0.bmp".format(filenames[j]), numpy.clip(i_degr, 0.0, 1.0))
    for k in range(len(lambdas[j])):
        u, _ = tv.opt_tv_admm(i_degr, sigma, lambdas[j][k], rhos[j][k], 125, 0.0e0, tv="hard")
        rt[j][0].append(skimage.measure.compare_psnr(i, u))
        rt[j][1].append(skimage.measure.compare_ssim(i, u))
        skimage.io.imsave("Figure5{}{}.bmp".format(filenames[j], k+1), numpy.clip(u, 0.0, 1.0))
        print("k = {} finished".format(k))
    print("{} finished".format(filenames[j]))


# In[6]:


with shelve.open("Result") as db:
    db["5filename"] = filenames
    db["5lambda"] = lambdas
    db["5rho"] = rhos
    db["5result"] = rt

