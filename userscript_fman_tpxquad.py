#!/usr/bin/env python3

'''
user script for making medipix matrices
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

from fitman import fitman

import os
import numpy as np

# filename
filenames =(
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-1101/pattern_d3_Npix0-20.txt',
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-1102/pattern_d3_Npix0-20.txt',
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-2113/pattern_d3_Npix0-20.txt')#,
#    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-1101/pattern_d3_Npix0-20.txt',
#    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-1102/pattern_d3_Npix0-20.txt',
#    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-2113/pattern_d3_Npix0-20.txt')

# library
libraries = (
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue646g26.2dl', # <-1101>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue567g29.2dl', # <-1102>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue725g28.2dl')#, # <-2113>
#    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue646g26.2dl',  # <-1101>
#    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue567g29.2dl',  # <-1102>
#    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue725g28.2dl')  # <-2113>13>

# fits
for i in range(len(filenames)):
    filename = filenames[i]
    library = libraries[i]
    basename, ext = os.path.splitext(filename)

    fm = fitman()
    fm.add_pattern(filename, library)
    P1 = np.array((0,))
    P2 = np.arange(0, 249) # 249
    fm.run_fits(P1, P2, method='chi2', get_errors=False, fit_sigma=True)

    fm.save_output(basename + '_fit.xls', save_figure=False)
    fm.save_output(basename + '_fit.csv', save_figure=True)
