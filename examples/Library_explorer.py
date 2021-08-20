#!/usr/bin/env python
# coding: utf-8

# # Library explorer
# 
# This is a short notebook to explore 2dl libraries.
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

#import sys
#sys.path.append("/home/eric/cernbox/PyCharm/PyFDD/")
import pyfdd
print('PyFDD version', pyfdd.__version__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from IPython.display import display

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)


# ## Import library
# 

# In[2]:


analysis_path = "../test_pyfdd/data_files"
lib_path = os.path.join(analysis_path, "sb600g05.2dl")
lib = pyfdd.Lib2dl(lib_path)
df = pd.DataFrame(data=lib.get_simulations_list(),
                  columns=["Spectrum number",
                           "Spectrum_description",
                           "factor",
                           "u2",
                           "sigma"])


# In[3]:


lib.print_header()


# In[4]:


display(df)


# ## Plot pattern
# 
# use the pattern number

# In[5]:


patt_number = 1


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
imgmat = lib.get_simulation_patt(patt_number)
plt.figure(dpi=150)
plt.contourf(imgmat)
plt.gca().set_aspect('equal')


# In[ ]:




