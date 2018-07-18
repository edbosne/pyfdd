
# coding: utf-8

# # Library explorer
# 
# This is a short notebook to explore 2dl libraries.
# 

# In[1]:

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import widgets, Label


import sys, os
sys.path.append("/home/eric/PycharmProjects/PyFDD")
from pyfdd import lib2dl


# ## Import library
# 

# In[10]:

analysis_path = "/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/"
lib_path = os.path.join(analysis_path, "FDD_libraries/GaN_89Sr/ue567g54.2dl")
lib = lib2dl(lib_path)
df = pd.DataFrame(data=lib.sim_list,
                  columns=["Spectrum number",
                           "Spectrum_description",
                           "factor",
                           "u2",
                           "sigma"])
#for entry in lib.sim_list:


# In[11]:

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1)
df


# ## Plot pattern
# 
# use the pattern number

# In[8]:

patt_number = 1


# In[9]:

get_ipython().magic('matplotlib inline')
imgmat = lib.get_simulation_patt(patt_number)
plt.contourf(imgmat)

