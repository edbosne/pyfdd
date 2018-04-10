
from PyFDD import lib2dl, PatternCreator, MedipixMatrix, fitman
from PyFDD.patterncreator import create_detector_mesh

import numpy as np
import matplotlib.pyplot as plt


def make_tpx_pattern(lib, patterns=(1,), name = 'temp_tpx.json'):

    xmesh, ymesh = create_detector_mesh(512, 512, 0.055, 300)
    # 5 subpixels is a good number for the pads
    gen = PatternCreator(lib, xmesh, ymesh, simulations=patterns, sub_pixels=1)

    fractions_per_sim = np.array([1, 0]) # site has zero occupancy, only background
    #fractions_per_sim /= fractions_per_sim.sum()
    total_events = 0.3 * 512**2
    pattern = gen.make_pattern(0.0, 0.0, 0, fractions_per_sim, total_events, sigma=0, type='montecarlo')
    print(pattern.sum())

    # create medipix matrix
    mm = MedipixMatrix(pattern_array=pattern)
    mm.manip_create_mesh(pixel_size=0.055, distance=300)

    return mm


if __name__ == '__main__':
    lib = lib2dl("/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/FDD_libraries/GaN_89Sr/ue567g54.2dl")
    mm = make_tpx_pattern(lib)

    fm = fitman()
    fm.add_pattern(mm, lib)
    P1 = np.arange(1, 249)  # last 248 set to 249
    fm.set_fixed_values(dx=0, dy=0, phi=0, sigma=0.1)  # pad=0.094, tpx=0.064
    fm.run_fits(P1, method='ml')
    fm.save_output('tpx_1site_fixed-orientation_test.csv', save_figure=False)
