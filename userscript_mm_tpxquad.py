#!/usr/bin/env python3

'''
user script for making medipix matrices
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'


from MedipixMatrix.MedipixMatrix import MedipixMatrix
import os
import matplotlib.pyplot as plt

# Create MedipixMatrix from file
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-1101/pattern_d3_Npix0-20.txt'
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-1102/pattern_d3_Npix0-20.txt'
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/RT/-2113/pattern_d3_Npix0-20.txt'
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-1101/pattern_d3_Npix0-20.txt'
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-1102/pattern_d3_Npix0-20.txt'
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/TPX/800C/-2113/pattern_d3_Npix0-20.txt'

basename, extention = os.path.splitext(filename)
mm2 = MedipixMatrix(file_path=filename, nChipsX=2, nChipsY=2, real_size=3)

# Manipulation methods
# -Orient
mm2.manip_orient('rr,mh')  # TimepixQuad orientation

# -Angular calibration
mm2.manip_create_mesh(pixel_size=0.055, distance=300)

# Zero central pix
mm2.zero_central_pix(0)

# Add extra pixels to account for bigger central pixels
mm2.manip_correct_central_pix()

# -Sum pixels, zero central pixels and remove edge pixels all in one
mm2.manip_compress(factor=2, rm_central_pix=0, rm_edge_pix=0)

# Smooth
# mm2.manip_smooth(2.0)

# -Mask pixels
mm2.mask_std(6)

# Plotting
f2 = plt.figure(2)
ax2 = plt.subplot('111')
mm2.draw(ax2, percentiles=(0.01, 0.99))

# get rectangle
# mm2.get_rectangle_tool()

# Measure angles - widget
mm2.get_angle_tool()

# Show
plt.show()

# close figure to continue

# print measured anlges
print('angle widget, center ', mm2.center, ', angle ', mm2.angle)

# mask array
mm2.mask_limits(limits=(mm2.center[0] - 2.8, mm2.center[0] + 2.8, mm2.center[1] - 2.8, mm2.center[1] + 2.8))

# save matrix
#mm2.io_save_json(basename + '_rebin16x16_180.json')
# ascii if to be used with fdd
# mm2.io_save_ascii('/home/eric/Desktop/test.txt')
