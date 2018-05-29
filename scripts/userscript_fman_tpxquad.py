#!/usr/bin/env python3

'''
user script for making medipix matrices
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

from PyFDD import fitman

import os
import numpy as np

folder = '/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/'

# filename
filenames2x2 =(
    '2015_GaN_24Na/TPX/RT/-1101/pattern_d3_Npix0-20_rebin2x2_180.json',
    '2015_GaN_24Na/TPX/RT/-1102/pattern_d3_Npix0-20_rebin2x2_180.json',
    '2015_GaN_24Na/TPX/RT/-2113/pattern_d3_Npix0-20_rebin2x2_180.json',
    '2015_GaN_24Na/TPX/800C/-1101/pattern_d3_Npix0-20_rebin2x2_180.json',
    '2015_GaN_24Na/TPX/800C/-1102/pattern_d3_Npix0-20_rebin2x2_180.json',
    '2015_GaN_24Na/TPX/800C/-2113/pattern_d3_Npix0-20_rebin2x2_180.json')

filenames16x16 =(
    '2015_GaN_24Na/TPX/RT/-1101/pattern_d3_Npix0-20_rebin16x16_180.json',
    '2015_GaN_24Na/TPX/RT/-1102/pattern_d3_Npix0-20_rebin16x16_180.json',
    '2015_GaN_24Na/TPX/RT/-2113/pattern_d3_Npix0-20_rebin16x16_180.json',
    '2015_GaN_24Na/TPX/800C/-1101/pattern_d3_Npix0-20_rebin16x16_180.json',
    '2015_GaN_24Na/TPX/800C/-1102/pattern_d3_Npix0-20_rebin16x16_180.json',
    '2015_GaN_24Na/TPX/800C/-2113/pattern_d3_Npix0-20_rebin16x16_180.json')

filenames22x22 =(
    '2015_GaN_24Na/TPX/RT/-1101/pattern_d3_Npix0-20_rebin22x22_180.json',
    '2015_GaN_24Na/TPX/RT/-1102/pattern_d3_Npix0-20_rebin22x22_180.json',
    '2015_GaN_24Na/TPX/RT/-2113/pattern_d3_Npix0-20_rebin22x22_180.json',
    '2015_GaN_24Na/TPX/800C/-1101/pattern_d3_Npix0-20_rebin22x22_180.json',
    '2015_GaN_24Na/TPX/800C/-1102/pattern_d3_Npix0-20_rebin22x22_180.json',
    '2015_GaN_24Na/TPX/800C/-2113/pattern_d3_Npix0-20_rebin22x22_180.json')


# library
libraries = (
    'FDD_libraries/GaN_24Na/ue646g26.2dl', # <-1101>
    'FDD_libraries/GaN_24Na/ue567g29.2dl', # <-1102>
    'FDD_libraries/GaN_24Na/ue725g28.2dl', # <-2113>
    'FDD_libraries/GaN_24Na/ue646g26.2dl',  # <-1101>
    'FDD_libraries/GaN_24Na/ue567g29.2dl',  # <-1102>
    'FDD_libraries/GaN_24Na/ue725g28.2dl')  # <-2113>13>

# fits

filenames = filenames2x2
#filenames = filenames16x16
#filenames = filenames22x22
for i in range(0, len(filenames)):
    filename = os.path.join(folder, filenames[i])
    library = os.path.join(folder, libraries[i])
    basename, ext = os.path.splitext(filename)

    print('Fitting ', filename, '\nwith ', library)

    fm = fitman()
    fm.set_pattern(filename, library)
    P1 = np.array((1,))
    #P2 = np.array((129,))
    P2 = np.arange(1, 249) # 249
    fm.set_fixed_values(sigma=0.064)  # pad=0.094, tpx=0.064
    fm.run_fits(P1, P2, cost_func='chi2', get_errors=False, sub_pixels=1)

    #fm.save_output(basename + '_2site-fit_subpix5.xls', save_figure=False)
    #fm.save_output(basename + '_2site(1,129)-fit-chi2_errors.csv', save_figure=False)
    fm.save_output(basename + '_2site-fit_subpix1_sigma0064.csv', save_figure=False)
