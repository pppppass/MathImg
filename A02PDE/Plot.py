
# coding: utf-8

# In[1]:


import shelve
import numpy
import matplotlib
matplotlib.use("pgf")
from matplotlib import pyplot


# In[2]:


with shelve.open("Result") as db:
    filename = db[str((1, 2, "filename"))]
    mu = db[str((1, 2, "mu"))]
    time = db[str((1, 2, "time"))]
    iter_ = db[str((1, 2, "iter"))]
    origpsnr = db[str((1, 2, "origpsnr"))]
    psnr = db[str((1, 2, "psnr"))]
    origssim = db[str((1, 2, "origssim"))]
    ssim = db[str((1, 2, "ssim"))]


# In[3]:


with open("Table11.tbl", "w") as f:
    for i in range(len(filename)):
        f.write("\\verb\"{}\" & {} & {:.3f} & {:.2e} & {:.2f} & {:.2f} \\\\\n".format(
            filename[i],
            iter_[i][1],
            time[i][1],
            iter_[i][1] / 10.0 / 256**2,
            iter_[i][1] / 10.0,
            numpy.sqrt(iter_[i][1] / 10.0)
        ))
        f.write("\\hline\n")
with open("Table12.tbl", "w") as f:
    for i in range(len(filename)):
        f.write("\\verb\"{}\" & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\\n".format(
            filename[i],
            origpsnr[i][1],
            psnr[i][1],
            psnr[i][1] - origpsnr[i][1],
            origssim[i][1],
            ssim[i][1],
            ssim[i][1] - origssim[i][1]
        ))
        f.write("\\hline\n")


# In[4]:


with shelve.open("Result") as db:
    filename = db[str((1, 3, "filename"))]
    mu = db[str((1, 3, "mu"))]
    iter_ = db[str((1, 3, "iter"))]
    psnr = db[str((1, 3, "psnr"))]
    ssim = db[str((1, 3, "ssim"))]


# In[5]:


for i in range(len(filename)):
    with open("Table2{}.tbl".format(i+1), "w") as f:
        iter__ = iter_[i]
        psnr_ = [e[2] for e in psnr if e[0] == filename[i]]
        ssim_ = [e[2] for e in ssim if e[0] == filename[i]]
        f.write("\\#Iterations & ")
        f.write("Degraded & " + "& ".join("{} ".format(iter__[j]) for j in range(len(iter__))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("$T$ & ")
        f.write("Degraded & " + "& ".join("{:.2e} ".format(iter__[j] / 200.0 / 256**2) for j in range(len(iter__))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("$T'$ & ")
        f.write("Degraded & " + "& ".join("{:.2e} ".format(iter__[j] / 200.0) for j in range(len(iter__))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("PSNR (\\Si{dB}) & ")
        f.write("& ".join("{:.5f} ".format(psnr_[j]) for j in range(len(psnr_))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("SSIM & ")
        f.write("& ".join("{:.5f} ".format(ssim_[j]) for j in range(len(ssim_))))
        f.write("\\\\\n")
        f.write("\\hline\n")


# In[6]:


with shelve.open("Result") as db:
    filename = db[str((2, 2, "filename"))]
    time = db[str((2, 2, "time"))]
    iter_ = db[str((2, 2, "iter"))]
    origpsnr = db[str((2, 2, "origpsnr"))]
    psnr = db[str((2, 2, "psnr"))]
    origssim = db[str((2, 2, "origssim"))]
    ssim = db[str((2, 2, "ssim"))]


# In[7]:


with open("Table31.tbl", "w") as f:
    for i in range(len(filename)):
        f.write("\\verb\"{}\" & {} & {:.3f} & {:.2e} & {:.2f} \\\\\n".format(
            filename[i],
            iter_[i][1],
            time[i][1],
            iter_[i][1] / 4.0 / 256**2,
            iter_[i][1] / 4.0
        ))
        f.write("\\hline\n")
with open("Table32.tbl", "w") as f:
    for i in range(len(filename)):
        f.write("\\verb\"{}\" & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\\n".format(
            filename[i],
            origpsnr[i][1],
            psnr[i][1],
            psnr[i][1] - origpsnr[i][1],
            origssim[i][1],
            ssim[i][1],
            ssim[i][1] - origssim[i][1]
        ))
        f.write("\\hline\n")


# In[8]:


with shelve.open("Result") as db:
    filename = db[str((2, 3, "filename"))]
    mu = db[str((2, 3, "mu"))]
    iter_ = db[str((2, 3, "iter"))]
    psnr = db[str((2, 3, "psnr"))]
    ssim = db[str((2, 3, "ssim"))]


# In[9]:


for i in range(len(filename)):
    with open("Table4{}.tbl".format(i+1), "w") as f:
        iter__ = iter_[i]
        psnr_ = [e[2] for e in psnr if e[0] == filename[i]]
        ssim_ = [e[2] for e in ssim if e[0] == filename[i]]
        f.write("\\#Iterations & ")
        f.write("Degraded & " + "& ".join("{} ".format(iter__[j]) for j in range(len(iter__))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("$T$ & ")
        f.write("Degraded & " + "& ".join("{:.2e} ".format(iter__[j] / 4.0 / 256**2) for j in range(len(iter__))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("$T'$ & ")
        f.write("Degraded & " + "& ".join("{:.2e} ".format(iter__[j] / 4.0) for j in range(len(iter__))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("PSNR (\\Si{dB}) & ")
        f.write("& ".join("{:.5f} ".format(psnr_[j]) for j in range(len(psnr_))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("SSIM & ")
        f.write("& ".join("{:.5f} ".format(ssim_[j]) for j in range(len(ssim_))))
        f.write("\\\\\n")
        f.write("\\hline\n")


# In[10]:


with shelve.open("Result") as db:
    filename = db[str((2, 4, "filename"))]
    time = db[str((2, 4, "time"))]
    iter_ = db[str((2, 4, "iter"))]
    origpsnr = db[str((2, 4, "origpsnr"))]
    psnr = db[str((2, 4, "psnr"))]
    origssim = db[str((2, 4, "origssim"))]
    ssim = db[str((2, 4, "ssim"))]


# In[11]:


with open("Table51.tbl", "w") as f:
    for i in range(len(filename)):
        f.write("\\verb\"{}\" & {} & {:.3f} & {:.2e} & {:.2f} \\\\\n".format(
            filename[i],
            iter_[i][1],
            time[i][1],
            iter_[i][1] / 4.0 / 256**2,
            iter_[i][1] / 4.0
        ))
        f.write("\\hline\n")
with open("Table52.tbl", "w") as f:
    for i in range(len(filename)):
        f.write("\\verb\"{}\" & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\\n".format(
            filename[i],
            origpsnr[i][1],
            psnr[i][1],
            psnr[i][1] - origpsnr[i][1],
            origssim[i][1],
            ssim[i][1],
            ssim[i][1] - origssim[i][1]
        ))
        f.write("\\hline\n")


# In[12]:


with shelve.open("Result") as db:
    filename = db[str((2, 5, "filename"))]
    mu = db[str((2, 5, "mu"))]
    iter_ = db[str((2, 5, "iter"))]
    psnr = db[str((2, 5, "psnr"))]
    ssim = db[str((2, 5, "ssim"))]


# In[13]:


for i in range(len(filename)):
    with open("Table6{}.tbl".format(i+1), "w") as f:
        iter__ = iter_[i]
        psnr_ = [e[2] for e in psnr if e[0] == filename[i]]
        ssim_ = [e[2] for e in ssim if e[0] == filename[i]]
        f.write("\\#Iterations & ")
        f.write("Degraded & " + "& ".join("{} ".format(iter__[j]) for j in range(len(iter__))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("$T$ & ")
        f.write("Degraded & " + "& ".join("{:.2e} ".format(iter__[j] / 4.0 / 256**2) for j in range(len(iter__))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("$T'$ & ")
        f.write("Degraded & " + "& ".join("{:.2e} ".format(iter__[j] / 4.0) for j in range(len(iter__))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("PSNR (\\Si{dB}) & ")
        f.write("& ".join("{:.5f} ".format(psnr_[j]) for j in range(len(psnr_))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("SSIM & ")
        f.write("& ".join("{:.5f} ".format(ssim_[j]) for j in range(len(ssim_))))
        f.write("\\\\\n")
        f.write("\\hline\n")


# In[14]:


with shelve.open("Result") as db:
    filename = db[str((2, 6, "filename"))]
    time = db[str((2, 6, "time"))]
    iter_ = db[str((2, 6, "iter"))]
    origpsnr = db[str((2, 6, "origpsnr"))]
    psnr = db[str((2, 6, "psnr"))]
    origssim = db[str((2, 6, "origssim"))]
    ssim = db[str((2, 6, "ssim"))]


# In[15]:


with open("Table71.tbl", "w") as f:
    for i in range(len(filename)):
        f.write("\\verb\"{}\" & {} & {:.3f} & {:.2e} & {:.2f} \\\\\n".format(
            filename[i],
            iter_[i][1],
            time[i][1],
            iter_[i][1] / 4.0 / 256**2,
            iter_[i][1] / 4.0
        ))
        f.write("\\hline\n")
with open("Table72.tbl", "w") as f:
    for i in range(len(filename)):
        f.write("\\verb\"{}\" & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\\n".format(
            filename[i],
            origpsnr[i][1],
            psnr[i][1],
            psnr[i][1] - origpsnr[i][1],
            origssim[i][1],
            ssim[i][1],
            ssim[i][1] - origssim[i][1]
        ))
        f.write("\\hline\n")


# In[16]:


with shelve.open("Result") as db:
    corr = db[str((2, 1, "corr"))]
    psnr = db[str((2, 1, "psnr"))]
    ssim = db[str((2, 1, "ssim"))]


# In[17]:


filename = ["tsukasa", "lena"]


# In[18]:


pyplot.figure(figsize=(8.0, 3.0))
for i in range(len(filename)):
    pyplot.subplot(1, 2, i+1)
    filename_ = filename[i]
    pyplot.plot([e[2] for e in corr if e[0] == filename_], label="$C_t$")
    pyplot.plot([e[2] / 30.0 for e in psnr if e[0] == filename_], label="<LABEL1~~~~>")
    pyplot.plot([e[2] for e in ssim if e[0] == filename_], label="SSIM")
    pyplot.xlabel("$m$")
    pyplot.ylabel("Value")
    pyplot.title("<LABEL2{}~~>".format(i+1))
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig("Figure12.pgf")
pyplot.show()


# In[19]:


with shelve.open("Result") as db:
    filename = db[str((3, 1, "filename"))]
    nu = db[str((3, 2, "nu"))]
    time = db[str((3, 1, "time"))]
    iter_ = db[str((3, 1, "iter"))]
    origpsnr = db[str((3, 1, "origpsnr"))]
    psnr = db[str((3, 1, "psnr"))]
    origssim = db[str((3, 1, "origssim"))]
    ssim = db[str((3, 1, "ssim"))]


# In[20]:


with open("Table81.tbl", "w") as f:
    for i in range(len(filename)):
        f.write("\\verb\"{}\" & {} & {:.3f} & {:.2e} \\\\\n".format(
            filename[i],
            iter_[i][1],
            time[i][1],
            iter_[i][1] / 10.0 / 256,
        ))
        f.write("\\hline\n")
with open("Table82.tbl", "w") as f:
    for i in range(len(filename)):
        f.write("\\verb\"{}\" & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\\n".format(
            filename[i],
            origpsnr[i][1],
            psnr[i][1],
            psnr[i][1] - origpsnr[i][1],
            origssim[i][1],
            ssim[i][1],
            ssim[i][1] - origssim[i][1]
        ))
        f.write("\\hline\n")


# In[21]:


with shelve.open("Result") as db:
    filename = db[str((3, 2, "filename"))]
    nu = db[str((3, 2, "nu"))]
    iter_ = db[str((3, 2, "iter"))]
    psnr = db[str((3, 2, "psnr"))]
    ssim = db[str((3, 2, "ssim"))]


# In[22]:


for i in range(len(filename)):
    with open("Table9{}.tbl".format(i+1), "w") as f:
        iter__ = iter_[i]
        psnr_ = [e[2] for e in psnr if e[0] == filename[i]]
        ssim_ = [e[2] for e in ssim if e[0] == filename[i]]
        f.write("\\#Iterations & ")
        f.write("Degraded & " + "& ".join("{} ".format(iter__[j]) for j in range(len(iter__))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("$T$ & ")
        f.write("Degraded & " + "& ".join("{:.2e} ".format(iter__[j] / 10.0 / 256) for j in range(len(iter__))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("PSNR (\\Si{dB}) & ")
        f.write("& ".join("{:.5f} ".format(psnr_[j]) for j in range(len(psnr_))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("SSIM & ")
        f.write("& ".join("{:.5f} ".format(ssim_[j]) for j in range(len(ssim_))))
        f.write("\\\\\n")
        f.write("\\hline\n")

