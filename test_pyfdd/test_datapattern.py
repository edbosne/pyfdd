

from pyfdd import DataPattern
import numpy as np
import matplotlib.pyplot as plt


def test_datapattern():
    print('Step by step example of using DataPattern')


    # Create DataPattern from array
    pattern = np.random.poisson(1000,(22,22))
    pattern[0, 0] = 0
    pattern[0, 1] = 0
    pattern[3, 0] = 0
    mm1 = DataPattern(pattern_array=pattern)
    f1 = plt.figure(1)
    ax1 = plt.subplot('111')
    mm1.draw(ax1)
    f1.show()
    mm1.io_save_json('jsontest.json')

    mm3 = DataPattern(file_path='jsontest.json')
    print('mm3 shape ', mm3.pattern_matrix.shape)
    f3 = plt.figure(3)
    ax3 = plt.subplot('111')
    mm3.draw(ax3)
    f3.show()


    # Create DataPattern from file
    filename = 'pattern_d3_Npix0-20.txt'
    mm2 = DataPattern(file_path=filename, nChipsX=2, nChipsY=2, real_size=3)

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

    # -Mask pixels
    mm2.mask_std(4, 0)

    # -Sum pixels, zero central pixels and remove edge pixels all in one
    mm2.manip_compress(factor=2, rm_central_pix=2, rm_edge_pix=4)

    # Load previously set mask
    #mm2.load_mask('test_masksave.txt')

    # Smooth
    #mm2.manip_smooth(2.0)

    # Save
    mm2.io_save_ascii('/home/eric/Desktop/test.txt')

    # Plotting
    f2 = plt.figure(2)
    ax2 = plt.subplot('111')
    mm2.draw(ax2, percentiles=(0.03, 0.99), blank_masked=True)

    # get rectangle
    #mm2.get_rectangle_tool()

    # Measure angles - widget
    mm2.get_angle_tool()

    # Show
    plt.show()
    plt.pause(5)

    # close figure to continue

    # print measured anlges
    print('angle widget, center ', mm2.center, ', angle ', mm2.angle)

    # mask array
    #mm2.mask_limits(limits=(mm2.center[0]-2.8, mm2.center[0]+2.8, mm2.center[1]-2.8, mm2.center[1]+2.8))
    mm2.set_fit_region(distance=2.5)

    f2 = plt.figure(2)
    ax2 = plt.subplot('111')
    mm2.draw(ax2, percentiles=(0.1, 0.95))
    mm2.get_rectangle_mask_tool()
    plt.show()
    plt.pause(5)

    f2 = plt.figure(2)
    ax2 = plt.subplot('111')
    mm2.draw(ax2, percentiles=(0.1, 0.95))
    plt.show(block=True)
    mm2.save_mask('test_masksave.txt')

    #mm2.io_save_json('/home/eric/Desktop/jsontest.json')
    #mm2.io_save_ascii('asciitest.txt', ignore_mask=True)

def test_assimetric():
    print('Step by step example of using DataPattern')

    # Create DataPattern from array
    pattern = np.random.poisson(1000,(100,30))
    mm1 = DataPattern(pattern_array=pattern)
    f1 = plt.figure(1)
    ax1 = plt.subplot('111')
    mm1.draw(ax1)
    f1.show()
    mm1.io_save_json('assimetric_test.json')

    mm2 = DataPattern(file_path='assimetric_test.json')
    print('mm2 shape ', mm2.pattern_matrix.shape)
    f2 = plt.figure(2)
    ax2 = plt.subplot('111')
    mm2.draw(ax2)
    f2.show()

    # Manipulation methods
    # -Orient
    mm2.manip_orient('rr,mh')  # TimepixQuad orientation

    # -Angular calibration
    mm2.manip_create_mesh(pixel_size=0.055, distance=300)
    #mm2.manip_create_mesh(pixel_size=1.4, distance=300)

    # -Sum pixels, zero central pixels and remove edge pixels all in one
    mm2.manip_compress(factor=4, rm_central_pix=0, rm_edge_pix=4)

    # Load previously set mask
    #mm2.load_mask('test_masksave.txt')

    # Smooth
    #mm2.manip_smooth(2.0)

    # Save
    #mm2.io_save_ascii('/home/eric/Desktop/test.txt')

    # Plotting
    f3 = plt.figure(3)
    ax3 = plt.subplot('111')
    mm2.draw(ax3, percentiles=(0, 1), blank_masked=True)

    # get rectangle
    #mm2.get_rectangle_tool()

    # Measure angles - widget
    mm2.get_angle_tool()

    # Show
    #plt.ion()
    plt.show()
    plt.pause(5)

    # close figure to continue

    # print measured anlges
    print('angle widget, center ', mm2.center, ', angle ', mm2.angle)

    # mask array
    #mm2.mask_limits(limits=(mm2.center[0]-2.8, mm2.center[0]+2.8, mm2.center[1]-2.8, mm2.center[1]+2.8))
    mm2.set_fit_region(distance=0.1)

    f4 = plt.figure(4)
    ax4 = plt.subplot('111')
    mm2.draw(ax4, percentiles=(0, 1))
    plt.show()



def test_compress():
    # one chip
    print('one chip')
    # 256 %16
    print('\n256 %16')
    pattern = np.random.poisson(1000,(256,256))
    mm = DataPattern(pattern_array=pattern)
    mm.manip_compress(factor=16, rm_central_pix=0, rm_edge_pix=0)

    # 256 %22
    print('\n256 %22')
    pattern = np.random.poisson(1000, (256, 256))
    mm = DataPattern(pattern_array=pattern)
    mm.manip_compress(factor=22, rm_central_pix=0, rm_edge_pix=0)

    # 516 %22
    print('\n516 %22')
    pattern = np.random.poisson(1000, (516, 516))
    mm = DataPattern(pattern_array=pattern)
    mm.manip_compress(factor=22, rm_central_pix=0, rm_edge_pix=0)

    # two chips
    print('\n\nTwo chips')
    # 512 %16
    print('\n512 %16')
    pattern = np.random.poisson(1000, (512, 512))
    mm = DataPattern(pattern_array=pattern, nChipsX=2, nChipsY=2)
    mm.manip_compress(factor=16, rm_central_pix=0, rm_edge_pix=0)

    # 512 %22
    print('\n516 %22')
    pattern = np.random.poisson(1000, (512, 512))
    mm = DataPattern(pattern_array=pattern, nChipsX=2, nChipsY=2, real_size=3)
    mm.manip_compress(factor=22, rm_central_pix=0, rm_edge_pix=0)



if __name__ == '__main__':
    test_datapattern()
    #test_compress()
    #test_assimetric()
