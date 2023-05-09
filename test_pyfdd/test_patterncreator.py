

import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

from pyfdd import Lib2dl, PatternCreator, DataPattern, FitManager, BkgPattern
from pyfdd.core.datapattern import create_detector_mesh


class TestPatternCreator(unittest.TestCase):

    def test_basics(self):

        lib = Lib2dl('data_files/sb600g05.2dl')
        #xmesh, ymesh = create_detector_mesh(22, 22, 1.4, 300)
        #xmesh, ymesh = create_detector_mesh(40, 40, 0.5, 300)
        xmesh, ymesh = create_detector_mesh(200, 200, 0.15, 300)
        gen = PatternCreator(lib, xmesh, ymesh, simulations=1, sub_pixels=5, mask_out_of_range=True)

        fractions_per_sim = 0.5
        total_events = 1e6
        pattern = gen.make_pattern(0.0, -0.1, -2, fractions_per_sim, total_events, sigma=0.1, pattern_type='yield')
        self.assertAlmostEqual(pattern.sum()/(200*200), 1, places=1)

        pattern = gen.make_pattern(0.0, -0.1, -2, fractions_per_sim, total_events, sigma=0.1, pattern_type='poisson')
        self.assertAlmostEqual(pattern.sum()/total_events, 1, places=1)

        pattern = gen.make_pattern(0.0, -0.1, -2, fractions_per_sim, total_events, sigma=0.1, pattern_type='montecarlo')
        self.assertAlmostEqual(pattern.sum()/total_events, 1, places=1)

    def test_pattern_with_background(self):

        # Setup background
        vertical_gradient = np.linspace(0.5, 1.5, 256)[np.newaxis]
        horizontal_gradient = np.linspace(0.8, 1.2, 256)
        patt_arr = ((np.ones((256, 256)) * horizontal_gradient) * vertical_gradient.T)
        bkg_patt = BkgPattern(pattern_array=patt_arr)
        bkg_patt.manip_create_mesh(0.055, 300)
        bkg_patt.set_sigma(1)

        # Set generator
        fractions_per_sim = 0.5
        total_events = 1e6
        lib = Lib2dl('data_files/sb600g05.2dl')
        xmesh, ymesh = create_detector_mesh(256, 256, 0.055, 300)
        gen = PatternCreator(lib, xmesh, ymesh, simulations=1, sub_pixels=5,
                             background_pattern= bkg_patt.get_smoothed_background(), background_factor=5,
                             mask_out_of_range=True)
        dp = gen.make_datapattern(0.0, -0.1, -2, fractions_per_sim, total_events, sigma=0.1, pattern_type='yield')
        plt.figure()
        dp.draw(plt.subplot(111))
        plt.show()



if __name__ == '__main__':
    unittest.main()
