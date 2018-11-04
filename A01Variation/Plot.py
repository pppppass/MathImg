
# coding: utf-8

# In[17]:


import shelve
import numpy


# In[2]:


with shelve.open("Result") as db:
    fn = db["1filename"]
    l = db["1lambda"]
    r = db["1rho"]
    rt = db["1result"]


# In[3]:


with open("Table11.tbl", "w") as f:
    for i in range(len(fn)):
        f.write("\\verb\"{}\" & {:.2e} & {:.2e} & {} & {:.3f} \\\\\n".format(fn[i], l[i], r[i], rt[1][i], rt[0][i]))
        f.write("\\hline\n")
with open("Table12.tbl", "w") as f:
    for i in range(len(fn)):
        f.write("\\verb\"{}\" & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\\n".format(fn[i], rt[2][i], rt[3][i], rt[3][i] - rt[2][i], rt[4][i], rt[5][i], rt[5][i] - rt[4][i]))
        f.write("\\hline\n")


# In[20]:


with shelve.open("Result") as db:
    fn = db["2filename"]
    l = db["2lambda"]
    r = db["2rho"]
    rt = db["2result"]


# In[23]:


for i in range(len(fn)):
    with open("Table2{}.tbl".format(i+1), "w") as f:
        f.write("$\\lambda$ & ")
        f.write("Degraded & " + "& ".join("{:.2e} ".format(l[i][j]) for j in range(len(l[i]))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("$\\rho$ & ")
        f.write("Degraded & " + "& ".join("{:.2e} ".format(r[i][j]) for j in range(len(l[i]))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("PSNR (\\Si{dB}) & ")
        f.write("& ".join("{:.5f} ".format(rt[i][0][j]) for j in range(len(l[i]) + 1)))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("SSIM & ")
        f.write("& ".join("{:.5f} ".format(rt[i][1][j]) for j in range(len(l[i]) + 1)))
        f.write("\\\\\n")
        f.write("\\hline\n")


# In[30]:


with shelve.open("Result") as db:
    fn = db["5filename"]
    l = db["5lambda"]
    r = db["5rho"]
    rt = db["5result"]


# In[31]:


for i in range(len(fn)):
    with open("Table5{}.tbl".format(i+1), "w") as f:
        l[i], r[i], rt[i][0][1:], rt[i][1][1:] = l[i][::-1], r[i][::-1], rt[i][0][-1:0:-1], rt[i][1][-1:0:-1]
        f.write("$\\lambda$ & ")
        f.write("Degraded & " + "& ".join("{:.2e} ".format(l[i][j]) for j in range(len(l[i]))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("$\\rho$ & ")
        f.write("Degraded & " + "& ".join("{:.2e} ".format(r[i][j]) for j in range(len(l[i]))))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("PSNR (\\Si{dB}) & ")
        f.write("& ".join("{:.5f} ".format(rt[i][0][j]) for j in range(len(l[i]) + 1)))
        f.write("\\\\\n")
        f.write("\\hline\n")
        f.write("SSIM & ")
        f.write("& ".join("{:.5f} ".format(rt[i][1][j]) for j in range(len(l[i]) + 1)))
        f.write("\\\\\n")
        f.write("\\hline\n")

