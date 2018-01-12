

from PyFDD import MedipixMatrix
import numpy as np
import matplotlib.pyplot as plt

def test_MedipixMatrix():
    print('Step by step example of using MedipixMatrix')

    # Create MedipixMatrix from array
    pattern = np.random.poisson(1000,(22,22))
    pattern[0, 0] = 0
    pattern[0, 1] = 0
    pattern[3, 0] = 0
    mm1 = MedipixMatrix(pattern_array=pattern)
    f1 = plt.figure(1)
    ax1 = plt.subplot('111')
    mm1.draw(ax1)
    f1.show()
    mm1.io_save_json('jsontest.json')

    mm3 = MedipixMatrix(file_path='jsontest.json')
    print('mm3 shape ', mm3.matrixCurrent.shape)
    f3 = plt.figure(3)
    ax3 = plt.subplot('111')
    mm3.draw(ax3)
    f3.show()

    # Create MedipixMatrix from file
    filename = 'pattern_d3_Npix0-20.txt'
    mm2 = MedipixMatrix(file_path=filename,nChipsX=2,nChipsY=2,real_size=3)

    # Manipulation methods
    # -Orient
    mm2.manip_orient('rr,mh')  # TimepixQuad orientation

    # -Angular calibration
    mm2.manip_create_mesh(pixel_size=0.055, distance=300)
    #mm2.manip_create_mesh(pixel_size=1.4, distance=300)

    # Zero central pix
    mm2.zero_central_pix(0)

    # Add extra pixels to account for bigger central pixels
    mm2.manip_correct_central_pix()

    # -Sum pixels, zero central pixels and remove edge pixels all in one
    mm2.manip_compress(factor=2, rm_central_pix=2, rm_edge_pix=4)

    # Smooth
    #mm2.manip_smooth(2.0)

    # -Mask pixels
    mm2.mask_std(6)

    # Save
    mm2.io_save_ascii('/home/eric/Desktop/test.txt')

    # Plotting
    f2 = plt.figure(2)
    ax2 = plt.subplot('111')
    mm2.draw(ax2, percentiles=(0.01, 0.99))

    # get rectangle
    #mm2.get_rectangle_tool()

    # Measure angles - widget
    mm2.get_angle_tool()

    # Show
    plt.show()

    # close figure to continue

    # print measured anlges
    print('angle widget, center ', mm2.center, ', angle ', mm2.angle)

    # mask array
    mm2.mask_limits(limits=(mm2.center[0]-2.8, mm2.center[0]+2.8, mm2.center[1]-2.8, mm2.center[1]+2.8))

    f2 = plt.figure(2)
    ax2 = plt.subplot('111')
    mm2.draw(ax2, percentiles=(0.01, 0.99))
    plt.show()
    #mm2.io_save_json('/home/eric/Desktop/jsontest.json')


if __name__ == '__main__':
    test_MedipixMatrix()