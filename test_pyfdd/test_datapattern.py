
import os
import unittest
import numpy as np

from pyfdd import DataPattern


class TestDataPattern(unittest.TestCase):

    def test_init_from_array(self):
        pattern = np.random.poisson(1000, (22, 22))
        pattern[0, 0] = 0
        pattern[0, 1] = 0
        pattern[3, 0] = 0
        dp = DataPattern(pattern_array=pattern)
        self.assertTrue(dp.pattern_matrix.size >= 0)

    def test_init_from_json(self):
        filepath = 'data_files/pad_dp_2M.json'
        dp_pad = DataPattern(file_path=filepath)
        self.assertTrue(dp_pad.pattern_matrix.size >= 0)

        filepath = 'data_files/tpx_quad_dp_2M.json'
        dp_timepix = DataPattern(file_path=filepath)
        self.assertTrue(dp_timepix.pattern_matrix.size >= 0)

    def test_init_from_txt(self):
        filepath = 'data_files/tpx_quad_array_2M.txt'
        dp_timepix = DataPattern(file_path=filepath, nChipsX=2, nChipsY=2, real_size=3)
        self.assertTrue(dp_timepix.pattern_matrix.size >= 0)

    def test_manipulations(self):

        # Get a timepix array
        filepath = 'data_files/tpx_quad_array_2M.txt'
        dp_timepix = DataPattern(file_path=filepath, nChipsX=2, nChipsY=2, real_size=3)
        self.assertTrue(dp_timepix.pattern_matrix.size >= 0)

        # Manipulation methods
        # -Orient
        dp_timepix.manip_orient('cw, mh')  # TimepixQuad orientation

        # -Angular calibration
        dp_timepix.manip_create_mesh(pixel_size=0.055, distance=300)
        #mm2.manip_create_mesh(pixel_size=1.4, distance=300)

        # Zero central pix
        dp_timepix.zero_central_pix(0)

        # Add extra pixels to account for bigger central pixels
        dp_timepix.manip_correct_central_pix()

        # -Mask pixels
        dp_timepix.mask_std(4, 0)

        # -Sum pixels, zero central pixels and remove edge pixels all in one
        dp_timepix.manip_compress(factor=2, rm_central_pix=2, rm_edge_pix=4)

        # Save and Load previously set mask
        dp_timepix.save_mask('test_masksave.txt')
        dp_timepix.load_mask('test_masksave.txt')

        if os.path.exists("test_masksave.txt"):
            os.remove("test_masksave.txt")

        # Save
        dp_timepix.io_save_ascii('test_ascii_save.txt')

        if os.path.exists("test_ascii_save.txt"):
            os.remove("test_ascii_save.txt")

        # Set angular resolution
        dx = -0.1
        dy = 0.2
        phi = 3
        dp_timepix.set_pattern_angular_pos(center=(dx, dy), angle=phi)

        # mask array
        dp_timepix.set_fit_region(distance=2.5)

    def test__update_compress_factors_tpx_quad(self):

        # Test remove central pixels update
        rm_central_pix, rm_edge_pix = \
            self.try__update_compress_factors_tpx_quad(factor=1, rm_central_pix=0, rm_edge_pix=0)
        self.assertEqual(rm_central_pix, 0)
        self.assertEqual(rm_edge_pix, 0)

        rm_central_pix, rm_edge_pix = \
            self.try__update_compress_factors_tpx_quad(factor=2, rm_central_pix=3, rm_edge_pix=0)
        self.assertEqual(rm_central_pix, 3)
        self.assertEqual(rm_edge_pix, 1)

        rm_central_pix, rm_edge_pix = \
            self.try__update_compress_factors_tpx_quad(factor=4, rm_central_pix=3, rm_edge_pix=0)
        self.assertEqual(rm_central_pix, 4)
        self.assertEqual(rm_edge_pix, 0)

        rm_central_pix, rm_edge_pix = \
            self.try__update_compress_factors_tpx_quad(factor=16, rm_central_pix=3, rm_edge_pix=0)
        self.assertEqual(rm_central_pix, 6)
        self.assertEqual(rm_edge_pix, 10)

        # Test prime factors
        self.try__update_compress_factors_tpx_quad(factor=3, rm_central_pix=3, rm_edge_pix=0)
        self.try__update_compress_factors_tpx_quad(factor=5, rm_central_pix=3, rm_edge_pix=0)
        self.try__update_compress_factors_tpx_quad(factor=7, rm_central_pix=3, rm_edge_pix=0)
        self.try__update_compress_factors_tpx_quad(factor=11, rm_central_pix=3, rm_edge_pix=0)

    def try__update_compress_factors_tpx_quad(self, factor, rm_central_pix, rm_edge_pix):

        # Get a timepix array
        filepath = 'data_files/tpx_quad_array_2M.txt'
        dp_timepix = DataPattern(file_path=filepath, nChipsX=2, nChipsY=2, real_size=3)
        dp_timepix.manip_correct_central_pix()

        rm_central_pix, rm_edge_pix = \
            dp_timepix._update_compress_factors_tpx_quad(factor, rm_central_pix, rm_edge_pix)

        # verify if the rest is zero
        nx, ny = dp_timepix.nx, dp_timepix.ny
        div_rest = (nx - 2 * rm_edge_pix) % factor
        self.assertEqual(div_rest, 0)
        div_rest = (ny - 2 * rm_edge_pix) % factor
        self.assertEqual(div_rest, 0)

        return rm_central_pix, rm_edge_pix

    def test_manip_compress(self):
        # one chip 256x256
        # factor 16
        pattern = np.random.poisson(1000, (256, 256))
        dp = DataPattern(pattern_array=pattern)
        dp.manip_compress(factor=16, rm_central_pix=0, rm_edge_pix=0)

        # factor 22
        pattern = np.random.poisson(1000, (256, 256))
        dp = DataPattern(pattern_array=pattern)
        dp.manip_compress(factor=22, rm_central_pix=0, rm_edge_pix=0)

        # one chip 516x516
        # factor 22
        pattern = np.random.poisson(1000, (512, 512))
        dp = DataPattern(pattern_array=pattern)
        dp.manip_compress(factor=22, rm_central_pix=0, rm_edge_pix=0)

        # two chips
        # factor 16
        pattern = np.random.poisson(1000, (516, 516))
        dp = DataPattern(pattern_array=pattern, nChipsX=2, nChipsY=2, real_size=3)
        dp.manip_compress(factor=16, rm_central_pix=0, rm_edge_pix=0)

        # factor 22
        pattern = np.random.poisson(1000, (516, 516))
        dp = DataPattern(pattern_array=pattern, nChipsX=2, nChipsY=2, real_size=3)
        dp.manip_compress(factor=22, rm_central_pix=0, rm_edge_pix=0)


if __name__ == '__main__':
    unittest.main()

