import os
import unittest
from unittest.mock import Mock, MagicMock

import matplotlib.pyplot as plt
import numpy as np

import pyfdd
from pyfdd import BkgPattern

class TestBkgPattern(unittest.TestCase):
    def test_bkgpattern_inicialization(self):
        patt_arr = np.ones((256, 256))
        bkg_patt = BkgPattern(pattern_array=patt_arr)
        self.assertIsInstance(bkg_patt, pyfdd.DataPattern)
        self.assertEqual(bkg_patt.smooth_sigma, 0)
        self.assertEqual(bkg_patt.correction_factor, 1)

    def test_calculate_correction_factor(self):
        patt_arr = np.ones((256, 256))
        bkg_patt = BkgPattern(pattern_array=patt_arr)
        bkg_patt.calculate_factor(bkg_time=2, data_time=1, data_cts=256*256)
        self.assertEqual(bkg_patt.correction_factor, 2)
        bkg_patt.set_factor(1.5)
        self.assertEqual(bkg_patt.correction_factor, 1.5)

    def test_generate_background(self):
        vertical_gradient = np.linspace(0.5,1.5,256)[np.newaxis]
        horizontal_gradient = np.linspace(0.8, 1.2, 256)
        patt_arr = ((np.ones((256, 256))*horizontal_gradient)*vertical_gradient.T)
        bkg_patt = BkgPattern(pattern_array=patt_arr)
        bkg_patt.manip_create_mesh(0.055, 300)
        bkg_patt.set_sigma(1)
        smoothed_background = bkg_patt.get_smoothed_background()

        plt.figure()
        bkg_patt.draw(plt.subplot(111))

        plt.figure()
        plt.imshow(smoothed_background)

        dp = bkg_patt.get_smoothed_background(as_datapattern=True)
        plt.figure()
        dp.draw(plt.subplot(111))
        plt.show()

        # Check if pattern have been calibrated and have a simular range
        # Smooth background
        # Interpolate points
        # Normalize
        pass

    def test_verify_shape(self):
        filepath = 'data_files/pad_dp_2M.json'
        dp_pad = pyfdd.DataPattern(file_path=filepath)

        patt_arr = np.ones((22, 22))
        bkg_patt = BkgPattern(pattern_array=patt_arr)
        self.assertTrue(bkg_patt.verify_shape(dp_pad))
        dp_pad.remove_edge_pixel(1)
        self.assertRaises(ValueError, bkg_patt.verify_shape, dp_pad)



    def test_verify_ranges(self):
        filepath = 'data_files/pad_dp_2M.json'
        dp_pad = pyfdd.DataPattern(file_path=filepath)

        patt_arr = np.ones((22, 22))
        bkg_patt = BkgPattern(pattern_array=patt_arr)
        self.assertRaises(ValueError, bkg_patt.verify_ranges, dp_pad)
        bkg_patt.manip_create_mesh(pixel_size=1.4, distance=310)
        self.assertTrue(bkg_patt.verify_ranges(dp_pad))



if __name__ == '__main__':
    unittest.main()
