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
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/RT/-1101/vat2857a_180.json',
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/RT/-1102/vat2859a_180.json',
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/RT/-2113/vat2861a_180.json',
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/800C/-1101/vat2865a_180.json',
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/800C/-1102/vat2864a_180.json',
    '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/800C/-2113/vat2863a_180.json')

# library
libraries = (
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue646g26.2dl', # <-1101>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue567g29.2dl', # <-1102>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue725g28.2dl', # <-2113>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue646g26.2dl',  # <-1101>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue567g29.2dl',  # <-1102>
    '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue725g28.2dl')  # <-2113>

# fits
for i in range(len(filenames)):
    filename = filenames[i]
    library = libraries[i]
    basename, ext = os.path.splitext(filename)

    print('Fitting ', filename, '\nwith ', library)

    fm = fitman()
    fm.add_pattern(filename, library)
    P1 = np.array((0,))
    P1 = np.arange(0, 249) # 249
    fm.run_fits(P1, method='chi2', get_errors=False, fit_sigma=True)

    fm.save_output(basename + '_1site-fit.xls', save_figure=False)
    fm.save_output(basename + '_1site-fit.csv', save_figure=False)