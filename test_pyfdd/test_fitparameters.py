import unittest

import numpy as np

import pyfdd
from pyfdd import FitParameters


class TestFitParameters(unittest.TestCase):

    def test_init_n_sites_should_raise_an_error_if_not_int(self):

        def n_sites_is_string():
            FitParameters(n_sites='1')

        self.assertRaises(ValueError, n_sites_is_string)

    def test_parameter_keys_after_init(self):
        parameter_keys_5 = ['dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3', 'f_p4', 'f_p5']

        for n_sites in range(5):
            fitp = FitParameters(n_sites=n_sites)
            self.assertEqual(fitp.get_keys(), parameter_keys_5[0:5 + n_sites])

    def test_parameter_bounds_after_init(self):
        parameter_bounds = FitParameters.default_bounds.copy()
        parameter_bounds.pop('f_px')
        parameter_bounds.update({'f_p1': (0, 1), 'f_p2': (0, 1), 'f_p3': (0, 1), 'f_p4': (0, 1), 'f_p5': (0, 1)})

        for n_sites in range(5):
            fitp = FitParameters(n_sites=n_sites)
            bounds_dict = fitp._bounds.copy()
            for key in bounds_dict:
                self.assertEqual(bounds_dict[key], parameter_bounds[key])

    def test_parameter_scale_after_init(self):
        parameter_scale = FitParameters.default_scale.copy()
        parameter_scale.pop('f_px')
        parameter_scale.update({'f_p1': 0.01, 'f_p2': 0.01, 'f_p3': 0.01, 'f_p4': 0.01, 'f_p5': 0.01})

        for n_sites in range(5):
            fitp = FitParameters(n_sites=n_sites)
            scale_dict = fitp.get_step_modifier()
            for key in scale_dict:
                self.assertEqual(scale_dict[key], parameter_scale[key])

    def test_initial_values_after_init(self):

        for n_sites in (1, 3, 5):
            fitp = FitParameters(n_sites=n_sites)
            initial_values = fitp.get_initial_values()
            fixed_values = fitp.get_fixed_values()
            site_str = f'f_p{n_sites}'
            self.assertGreaterEqual(0.15, initial_values[site_str])
            self.assertEqual(fixed_values[site_str], False)
            self.assertEqual(len(initial_values), 5 + n_sites)

    def test_update_initial_values_with_datapattern(self):
        # Dummy DataPattern
        patt = np.ones((100, 100))
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        x_mesh, y_mesh = np.meshgrid(x, y)

        dp = pyfdd.DataPattern(pattern_array=patt)
        dp.set_xymesh(xmesh=x_mesh, ymesh=y_mesh)
        dp.set_pattern_angular_pos((-0.1, 0.5), 3)

        # Test
        fitp = FitParameters(n_sites=1)
        fitp.update_initial_values_with_datapattern(datapattern=dp)
        initial_values = fitp.get_initial_values()

        self.assertEqual(initial_values['dx'], -0.1)
        self.assertEqual(initial_values['dy'], 0.5)
        self.assertEqual(initial_values['phi'], 3)
        self.assertEqual(initial_values['total_cts'], dp.pattern_matrix.sum())

    def test_update_bounds_with_datapattern(self):
        patt = np.ones((100, 100))
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-0.5, 3.5, 100)
        x_mesh, y_mesh = np.meshgrid(x, y)

        dp = pyfdd.DataPattern(pattern_array=patt)
        dp.set_xymesh(xmesh=x_mesh, ymesh=y_mesh)

        fp = FitParameters(n_sites=1)
        fp.update_bounds_with_datapattern(datapattern=dp)
        result_bounds = fp.get_bounds()

        # Assumes default bounds are (-3, 3)
        self.assertEqual(result_bounds['dx'], (-2, 2))
        self.assertEqual(result_bounds['dy'], (-0.5, 3))

    def test_change_initial_values(self):
        fp = FitParameters(n_sites=2)
        fp.change_initial_values(dx=0.1, f_p1=0.1)
        initial_values = fp.get_initial_values()

        self.assertEqual(initial_values['dx'], 0.1)
        self.assertEqual(initial_values['dy'], FitParameters.default_initial_values['dy'])  # Default 0
        self.assertEqual(initial_values['f_p1'], 0.1)
        self.assertEqual(initial_values['f_p2'], FitParameters.default_initial_values['f_px'])  # Default 0.15

    def test_change_initial_values_unfixes_parameter(self):
        fp = FitParameters(n_sites=1)

        fixed_values = fp.get_fixed_values()
        self.assertEqual(fixed_values['sigma'], True)
        self.assertEqual(fixed_values['total_cts'], True)

        fp.change_initial_values(sigma=0.2, total_cts=1)
        initial_values = fp.get_initial_values()
        fixed_values = fp.get_fixed_values()
        self.assertEqual(fixed_values['sigma'], False)
        self.assertEqual(fixed_values['total_cts'], False)
        self.assertEqual(initial_values['sigma'], 0.2)
        self.assertEqual(initial_values['total_cts'], 1)

    def test_change_fixed_values(self):
        fp = FitParameters(n_sites=1)

        fixed_values = fp.get_fixed_values()
        self.assertEqual(fixed_values['sigma'], True)
        self.assertEqual(fixed_values['total_cts'], True)

        fp.change_fixed_values(sigma=False, total_cts=False)
        fixed_values = fp.get_fixed_values()
        self.assertEqual(fixed_values['sigma'], False)
        self.assertEqual(fixed_values['total_cts'], False)

        fp.change_fixed_values(sigma=0.1, total_cts=1)
        fixed_values = fp.get_fixed_values()
        self.assertEqual(fixed_values['sigma'], True)
        self.assertEqual(fixed_values['total_cts'], True)
        initial_values = fp.get_initial_values()
        self.assertEqual(initial_values['sigma'], 0.1)
        self.assertEqual(initial_values['total_cts'], 1)

    def test_change_bounds(self):
        fp = FitParameters(n_sites=1)
        fp.change_bounds(dx=(-1, 1))
        bounds = fp.get_bounds()
        self.assertEqual(bounds['dx'], (-1, 1))
        self.assertEqual(bounds['dy'], FitParameters.default_bounds['dy'])  # (-3, +3)

    def test_change_step_modifier(self):
        fp = FitParameters(n_sites=1)
        fp.change_step_modifier(dx=1)
        scale = fp.get_step_modifier()
        self.assertEqual(scale['dx'], 1)
        self.assertEqual(scale['dy'], FitParameters.default_scale['dy'])  # .01

    def test__str__(self):
        fp = FitParameters(n_sites=1)
        print(fp)

    def test_get_p0_pfix(self):
        fp = FitParameters(n_sites=1)
        keys = fp.get_keys()

        p0, pfix = fp.get_p0_pfix()
        self.assertIsInstance(p0, tuple)
        self.assertIsInstance(pfix, tuple)
        self.assertEqual(len(p0), len(keys))
        self.assertEqual(len(pfix), len(keys))

    def test_get_scale_for_fit(self):
        patt = np.ones((100, 100)) * 100  # 10e6
        total_cts = patt.sum()

        dp = pyfdd.DataPattern(pattern_array=patt)

        fp = FitParameters(n_sites=1)
        fp.update_initial_values_with_datapattern(datapattern=dp)
        keys = fp.get_keys()

        scale = fp.get_scale_for_fit(cost_function='chi2')
        for k, scale_val in zip(keys, scale):
            if k == 'total_cts':
                self.assertEqual(scale_val, total_cts * FitParameters.default_scale['total_cts'])

        scale = fp.get_scale_for_fit(cost_function='ml')
        for k, scale_val in zip(keys, scale):
            if k == 'total_cts':
                self.assertEqual(scale_val, -1)


if __name__ == '__main__':
    unittest.main()
