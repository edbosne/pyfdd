
# coding: utf-8

# # Two sites fits
# 
# In this notebook I simply paste a real example so that you can see how to use the fitmanager in a loop.
# 

# In[ ]:

import pyfdd
print('PyFDD version', pyfdd.__version__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1)


# In[ ]:

analysis_path = "/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/"

axis_names = (
    "[-1101]",
    "[-1101]",
    "[-1101]",
    "[-1101]",
    "[-1101]",
    "[-1101]")

pattern_filenames = (
    "2016_GaN_27Mg/TPX/lowfluence/RT/pattern_d3_Npix0-50_rebin8x8.json",
    "2016_GaN_27Mg/TPX/lowfluence/600C/pattern_d3_Npix0-50_rebin8x8.json",
    "2016_GaN_27Mg/TPX/lowfluence/800C/pattern_d3_Npix0-50_rebin8x8.json",
    "2016_GaN_27Mg/TPX/highfluence/RT/pattern_d3_Npix0-50_rebin8x8.json",
    "2016_GaN_27Mg/TPX/highfluence/600C/pattern_d3_Npix0-50_rebin8x8.json",
    "2016_GaN_27Mg/TPX/highfluence/800C/pattern_d3_Npix0-50_rebin8x8.json")


library_names = (
    "FDD_libraries/2016_GaN_27Mg/ue650g23.2dl",) #-1101


# In[ ]:

get_ipython().run_cell_magic('time', '', 'for i in range(0, len(axis_names)):\n   \n    # path operations do join the filenames with the folder path\n    filename = os.path.join(analysis_path, pattern_filenames[i])\n    library = os.path.join(analysis_path, library_names[0])\n    basename, ext = os.path.splitext(filename)\n    dirname = os.path.dirname(filename)\n    \n    print(\'------//------//------\')\n    print(\'Fitting: \\n\', filename, \'\\nwith: \\n \', library)\n\n    # Initiate the fitmanager and inform it of the method to use and how many subpixels\n    # The number of subpixels used in the past has been 6 for the pad\n    # for the timepix depending on the number of added pixels in the DataPattern \n    # is set to half the number of added pixels\n    fm = pyfdd.FitManager(cost_function=\'chi2\', n_sites=2, sub_pixels=2)\n    \n    # Set the pattern and library to fit with\n    fm.set_pattern(filename, library)\n    \n    # Set a fixed value if needed \n    fm.set_fixed_values(sigma=0.06)\n    \n    # You probably wont need to change defaults but in any case the methods are here\n    # Change default bounds\n    #fm.set_bounds()\n    \n    # Change default  \n    #fm.set_step_modifier(dx=.01,dy=.01,phi=.10,sigma=.001,total_cts=0.01,f_p1=.01,f_p2=.01)\n    \n    # Change initial values\n    #fm.set_initial_values()\n    \n    # Set a minization profile\n    # use \'coarse\' for testing (it can avoid the program to hang if the fit is too hard),\n    # \'default\' for standard\n    # \'fine\' for when you are sure the fit will produce a result and want a smoother function\n    fm.set_minimization_settings(profile=\'fine\')\n    \n    # Set a pattern or range of patterns to fit\n    P1 = np.array([1])   # single site\n    P2 = np.arange(1,249) # range - one more than the last\n    \n    # Run fits \n    # remember to set get_errors to True if you want them. This increases the fit time.\n    fm.run_fits(P1, P2, get_errors=True)\n    \n    # Choose a smart name and save the results\n    save_name = dirname + "/" + axis_names[i] + \'_2site-fit_chi2-fine_sigma=0.06.csv\'\n    print(\'saving to\\n\', save_name)\n    fm.save_output(save_name, save_figure=False)')


# ## Visualizing a single fit
# 
# For visualizing a single fit the procedure is the same but instead of using
# 
#     fm.run_fits(P1, P2, get_errors=True)
#     
# use a scalars for P1, P2 and P3 and run as,
# 
#     fm.run_single_fit(P1, P2, verbose=1, verbose_graphics=True, get_errors=True)
# 

# ## Visualizing results
# 
# The results table is accessible in `FitManager.df`.
# 
# The results pattern can be accessed with 
# ```
# FitManager.get_pattern_from_last_fit(normalization=None):
# # or
# FitManager.get_pattern_from_best_fit(normalization=None):
# ```
# 
# The normalization can be `counts, yield or probability`
# 

# In[ ]:

fm.df


# In[ ]:

sim_patt = fm.get_pattern_from_last_fit(normalization='yield')

fg = plt.figure()
ax = fg.add_subplot('111')
sim_patt.draw(ax, percentiles=(0.01, 0.99), plot_type='contour')

