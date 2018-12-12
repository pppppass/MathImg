
# coding: utf-8

# In[9]:


import time
import shelve
import numpy
import scipy.io
import surf


# In[10]:


filename = "lucy"


# In[11]:


m = scipy.io.loadmat("dataset/{}.mat".format(filename))


# In[12]:


f = m["f"]
phi = m["phi"]

# f = m["f"][::2, ::2, ::2]
# phi = m["phi"][::2, ::2, ::2]


# In[5]:


n = (240, 144, 396)

# n = (120, 72, 198)


# In[6]:


start = time.time()
u = surf.evolve_surf(f, n, phi, 50.0, 75.0, "haar", 1, 100, debug=True)
end = time.time()


# In[7]:


scipy.io.savemat("Result1{}.mat".format(filename), {"u": u})


# In[8]:


with shelve.open("Result") as db:
    db[str((1, filename, "time"))] = end - start
    db[str((1, filename, "size"))] = n

