#!/usr/bin/env python3

'''
user script for making medipix matrices
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'


from MedipixMatrix.MedipixMatrix import MedipixMatrix
import os
import matplotlib.pyplot as plt
import numpy as np

# Create MedipixMatrix from file
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/RT/-1101/vat2857a.2db'
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/RT/-1102/vat2859a.2db'
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/RT/-2113/vat2861a.2db'
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/800C/-1101/vat2865a.2db'
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/800C/-1102/vat2864a.2db'
filename = '/home/eric/cernbox/Channeling_analysis/2015_GaN_24Na/PAD/800C/-2113/vat2863a.2db'
basename, extention = os.path.splitext(filename)
mm2 = MedipixMatrix(file_path=filename)

# Manipulation methods
# -Orient
mm2.manip_orient('rr')  # PAD4 orientation

# -Mask pixels
mask = np.zeros(mm2.matrixOriginal.shape)
mask[1,1]   = mask[1,2]   = mask[1,20]  = mask[2,1]  = mask[9,11]  = mask[9,20]  = mask[10,10] = mask[10,11] =\
mask[10,12] = mask[10,19] = mask[10,20] = mask[11,2] = mask[11,11] = mask[11,20] = mask[12,1]  = mask[12,2]  = \
mask[12,3]  = mask[13,1]  = mask[13,2]  = mask[13,3] = mask[14,2]  = mask[20,1]  = mask[20,2]  = mask[20,20] = 1
mm2.set_mask(mask)

# -Angular calibration
mm2.manip_create_mesh(pixel_size=1.4, distance=300)

# -Sum pixels, zero central pixels and remove edge pixels all in one
mm2.manip_compress(factor=1, rm_central_pix=0, rm_edge_pix=1)

# Smooth
# mm2.manip_smooth(2.0)

#mm2.mask_std(6)

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
f2 = plt.figure(2)
ax2 = plt.subplot('111')
mm2.draw(ax2, percentiles=(0.01, 0.99))
plt.show()
mm2.io_save_json(basename + '_180.json')
# ascii if to be used with fdd
# mm2.io_save_ascii('/home/eric/Desktop/test.txt')
