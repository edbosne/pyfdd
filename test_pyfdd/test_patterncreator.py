

import os
import unittest
import numpy as np

from pyfdd import Lib2dl, PatternCreator, DataPattern, FitManager
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


if __name__ == '__main__':
    unittest.main()
