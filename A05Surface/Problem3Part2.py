
# coding: utf-8

# In[8]:


import time
import shelve
import numpy
import scipy.io
import surf


# In[9]:


filename = "cndragon"


# In[10]:


m = scipy.io.loadmat("dataset/{}.mat".format(filename))


# In[11]:


f = m["f"]
phi = m["phi"]

# f = m["f"][::2, ::2, ::2]
# phi = m["phi"][::2, ::2, ::2]


# In[12]:


n = (320, 188, 220)

# n = (160, 94, 110)


# In[13]:


start = time.time()
u = surf.evolve_surf(f, n, phi, 50.0, 75.0, "haar", 1, 100, debug=True)
end = time.time()


# In[14]:


scipy.io.savemat("Result1{}.mat".format(filename), {"u": u})


# In[15]:


with shelve.open("Result") as db:
    db[str((1, filename, "time"))] = end - start
    db[str((1, filename, "size"))] = n

