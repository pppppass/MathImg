
# coding: utf-8

# In[1]:


import shelve
import numpy
import skimage.io
from utils import degrade


# In[2]:


filename_list = ["lena", "tsukasa"]
eta = 20.0 / 255.0
mu = 1.0 / 4.0
k = 100.0
iter_ = 100


# In[3]:


rt = [[], [], []]


# In[4]:


def nonlinear(g):
    b = 1.0 / (1.0 + g / k)
    return b


# In[5]:


def grad_x_2(u):
    g = numpy.zeros((n_x - 1, n_y))
    g[:, 1:-1] = (u[:-1, 2:] + u[1:, 2:] - u[:-1, :-2] - u[1:, :-2]) / h / 4.0
    g[:, 0] = (u[:-1, 1] + u[1:, 1] - u[:-1, 0] - u[1:, 0]) / h / 4.0
    g[:, -1] = (u[:-1, -1] + u[1:, -1] - u[:-1, -2] - u[1:, -2]) / h / 4.0
    g = g**2 + ((u[1:, :] - u[:-1, :]) / h)**2
    return g
def grad_y_2(u):
    g = numpy.zeros((n_x, n_y - 1))
    g[1:-1, :] = (u[2:, :-1] + u[2:, 1:] - u[:-2, :-1] - u[:-2, 1:]) / h / 4.0
    g[0, :] = (u[1, :-1] + u[1, 1:] - u[0, :-1] - u[0, 1:]) / h / 4.0
    g[-1, :] = (u[-1, :-1] + u[-1, 1:] - u[-2, :-1] - u[-2, 1:]) / h / 4.0
    g = g**2 + ((u[:, 1:] - u[:, :-1]) / h)**2
    return g


# In[6]:


for filename in filename_list:
    
    i = skimage.io.imread("dataset/{}.bmp".format(filename)) / 255.0
    i_degr = degrade.calc_degrade(i, None, eta)
    
    n_x, n_y = i.shape
    h = 1.0 / min(n_x, n_y)
    tau = mu * h**2
    
    u = i_degr
    
    for it in range(iter_):
        
        u_ = numpy.zeros(u.shape)
        b = nonlinear(grad_x_2(u))
        u_[:-1, :] += (u[:-1, :] - u[1:, :]) * b
        u_[1:, :] += (u[1:, :] - u[:-1, :]) * b
        b = nonlinear(grad_y_2(u))
        u_[:, :-1] += (u[:, :-1] - u[:, 1:]) * b
        u_[:, 1:] += (u[:, 1:] - u[:, :-1]) * b
        u = u - mu * u_
        u = u - mu * u_
    
        rt[0].append((filename, it, numpy.corrcoef((i_degr - u).flatten(), u.flatten())[0, 1]))
        rt[1].append((filename, it, skimage.measure.compare_psnr(i, u)))
        rt[2].append((filename, it, skimage.measure.compare_ssim(i, u)))
    
    print("{} finished".format(filename))


# In[7]:


with shelve.open("Result") as db:
    db[str((2, 1, "corr"))] = rt[0]
    db[str((2, 1, "psnr"))] = rt[1]
    db[str((2, 1, "ssim"))] = rt[2]

