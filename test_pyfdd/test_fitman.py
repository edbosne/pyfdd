
import os
import unittest
import numpy as np

from pyfdd import Lib2dl, PatternCreator, DataPattern, FitManager


class TestFitManager(unittest.TestCase):

    def test_basics(self):
        lib = Lib2dl('data_files/sb600g05.2dl')
        dp_pad = DataPattern('data_files/pad_dp_2M.json')

        fm = FitManager(cost_function='chi2', n_sites=2, sub_pixels=8)
        fm.set_pattern(dp_pad, lib)
        fm.set_fixed_values(sigma=0.05)  # pad=0.094, tpx=0.064
        fm.set_bounds(phi=(-20,20))
        fm.set_initial_values(phi=0.5)
        fm.set_minimization_settings(profile='fine')

        p1 = 1
        p2 = 23
        fm.run_single_fit(p1, p2, verbose_graphics=False)

        self.assertAlmostEqual(fm.df_horizontal['x'][0], -0.1, places=2)
        self.assertAlmostEqual(fm.df_horizontal['y'][0], 0.2, places=2)


if __name__ == '__main__':
    unittest.main()
