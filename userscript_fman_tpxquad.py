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
filenames2x2 =(
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-1101/pattern_d3_Npix0-20_rebin2x2_180.json',
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-1102/pattern_d3_Npix0-20_rebin2x2_180.json',
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-2113/pattern_d3_Npix0-20_rebin2x2_180.json')
#    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-1101/pattern_d3_Npix0-20_rebin2x2_180.json',
#    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-1102/pattern_d3_Npix0-20_rebin2x2_180.json',
#    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-2113/pattern_d3_Npix0-20_rebin2x2_180.json')

# filenames16x16 =(
#     '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-1101/pattern_d3_Npix0-20_rebin16x16_180.json',
#     '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-1102/pattern_d3_Npix0-20_rebin16x16_180.json',
#     '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-2113/pattern_d3_Npix0-20_rebin16x16_180.json',
#     '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-1101/pattern_d3_Npix0-20_rebin16x16_180.json',
#     '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-1102/pattern_d3_Npix0-20_rebin16x16_180.json',
#     '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-2113/pattern_d3_Npix0-20_rebin16x16_180.json')

# library
libraries = (
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue646g26.2dl', # <-1101>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue567g29.2dl', # <-1102>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue725g28.2dl', # <-2113>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue646g26.2dl',  # <-1101>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue567g29.2dl',  # <-1102>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue725g28.2dl')  # <-2113>13>

# fits

filenames = filenames2x2
#filenames = filenames16x16
for i in range(0, len(filenames)):
    filename = filenames[i]
    library = libraries[i]
    basename, ext = os.path.splitext(filename)

    print('Fitting ', filename, '\nwith ', library)

    fm = fitman()
    fm.add_pattern(filename, library)
    P1 = np.array((0,))
    P2 = np.array((128,))
    P3 = np.arange(0, 249) # 249
    fm.run_fits(P1, P2, P3, method='chi2', get_errors=False, fit_sigma=True)

    fm.save_output(basename + '_3site-fit.xls', save_figure=False)
    fm.save_output(basename + '_3site-fit.csv', save_figure=True)
