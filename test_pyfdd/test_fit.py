
import os
import unittest
import numpy as np
import math

from pyfdd import Lib2dl, Fit, DataPattern, PatternCreator, BkgPattern
from pyfdd.core.datapattern import create_detector_mesh


class TestFit(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Runs when class is initiated
        cls.lib = Lib2dl('data_files/sb600g05.2dl')
        cls.sites = (1, 23)

    def setUp(self) -> None:
        # Runs before each test
        dp_pad = DataPattern('data_files/pad_dp_2M.json')
        self.xmesh = dp_pad.xmesh
        self.ymesh = dp_pad.ymesh
        self.patt = dp_pad.pattern_matrix

    def test_chi2_fit(self):
        ft = Fit(self.lib, self.sites)

        # set a fitting routine
        counts_ordofmag = 10 ** (int(math.log10(self.patt.sum())))
        ft.set_data_pattern(self.xmesh, self.ymesh, self.patt)
        ft._parameters_dict['sub_pixels']['value'] = 1

        ft.set_scale_values(dx=1, dy=1, phi=1, total_cts=counts_ordofmag, sigma=0.1, f_p1=1)
        ft.set_initial_values(0, 0, 0, counts_ordofmag, sigma=0.1)
        # ft.set_inicial_values(mm.center[0], mm.center[1], mm.angle, counts_ordofmag, sigma=0.1)
        # ft.fix_parameters(dx=False, dy=False, phi=False, total_cts=False, sigma=False, f_p1=False, f_p2=False)

        ft.minimize_chi2()
        self.assertAlmostEqual(ft.results['x'][0], -0.1, places=2)
        self.assertAlmostEqual(ft.results['x'][1], 0.2, places=2)
        print(ft.results)
        print('sigma in sim step units - ', ft.results['x'][4] / self.lib.xstep)
        print('Calculating errors ...')
        # var = ft.get_variance_from_hessian(ft.results['x'], enable_scale=False, func='chi_square')
        # print('var - ', var)
        # ft.print_variance(ft.res['x'],var)
        # x = res['x'] * ft.p0_scale[0:5]
        # ft.set_scale_values()
        # # There is a warning because the hessian starts with a step too big, don't worry about it
        # H = nd.Hessian(ft.log_likelihood_call)#,step=1e-9)
        # hh = H(x)
        # print(hh)
        # print(np.linalg.inv(hh))
        # ft.set_scale_values(dx=1, dy=1, phi=10, f_rand=counts_ordofmag, f_p1=counts_ordofmag)
        # ft.print_results(res,hh)

        print('data points ', np.sum(~self.patt.mask))

        return ft

    def test_ml_fit(self):
        ft = Fit(self.lib, self.sites)
        ft.verbose_graphics = False

        # set a fitting routine
        counts_ordofmag = 10 ** (int(math.log10(self.patt.sum())))
        ft.set_data_pattern(self.xmesh, self.ymesh, self.patt)
        # ft.set_patterns_to_fit(249-249,377-249)
        # ft._set_patterns_to_fit(1, 65)  # ,129)
        ft._parameters_dict['sub_pixels']['value'] = 1

        ft.set_scale_values(dx=1, dy=1, phi=1, total_cts=-1, f_p1=1, f_p2=1)
        ft.set_initial_values(0, 0, 0, -1, sigma=0.1)
        ft.fix_parameters(dx=False, dy=False, phi=False, total_cts=False, sigma=False, f_p1=False, f_p2=False)
        # ft.set_inicial_values(mm.center[0], mm.center[1], mm.angle, -1, sigma=0.1)
        ft.maximize_likelyhood()
        self.assertAlmostEqual(ft.results['x'][0], -0.1, places=2)
        self.assertAlmostEqual(ft.results['x'][1], 0.2, places=2)
        print(ft.results)
        print('sigma in sim step units - ', ft.results['x'][4] / self.lib.xstep)
        print('Calculating errors ...')
        # var = ft.get_variance_from_hessian(ft.results['x'], enable_scale=False, func='likelihood')
        # print('var - ', var)
        # ft.print_variance(ft.res['x'],var)
        # ft.get_location_errors(res['x'], (0,), last=300, func='likelihood')$

        print('data points ', np.sum(~self.patt.mask))

    def test_run_fit_with_background(self):
        # Setup background
        npixels = 22
        vertical_gradient = np.linspace(0.5, 1.5, npixels)[np.newaxis]
        horizontal_gradient = np.linspace(0.8, 1.2, npixels)
        patt_arr = ((np.ones((npixels, npixels)) * horizontal_gradient) * vertical_gradient.T)
        bkg_patt = BkgPattern(pattern_array=patt_arr)
        bkg_patt.manip_create_mesh(1.4, 300)
        bkg_patt.set_sigma(1)

        ft = Fit(self.lib, self.sites)
        ft.verbose_graphics = False

        # set a fitting routine
        counts_ordofmag = 10 ** (int(math.log10(self.patt.sum())))
        ft.set_data_pattern(self.xmesh, self.ymesh, self.patt, background_pattern=bkg_patt.get_smoothed_background(),
                            background_factor=1.01)
        # ft.set_patterns_to_fit(249-249,377-249)
        # ft._set_patterns_to_fit(1, 65)  # ,129)
        ft._parameters_dict['sub_pixels']['value'] = 1

        ft.set_scale_values(dx=1, dy=1, phi=1, total_cts=-1, f_p1=1, f_p2=1)
        ft.set_initial_values(0, 0, 0, -1, sigma=0.1)
        ft.fix_parameters(dx=False, dy=False, phi=False, total_cts=False, sigma=False, f_p1=False, f_p2=False)
        # ft.set_inicial_values(mm.center[0], mm.center[1], mm.angle, -1, sigma=0.1)
        ft.maximize_likelyhood()
        self.assertAlmostEqual(ft.results['x'][0], -0.1, places=2)
        self.assertAlmostEqual(ft.results['x'][1], 0.2, places=2)
        print(ft.results)
        print('sigma in sim step units - ', ft.results['x'][4] / self.lib.xstep)


if __name__ == '__main__':
    unittest.main()
