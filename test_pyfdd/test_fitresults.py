import os
import unittest
from unittest.mock import Mock, MagicMock

import pyfdd
from pyfdd import FitResults, FitManager, Lib2dl, DataPattern


class TestFitResults(unittest.TestCase):
    def test_initialization_of_fitresults(self):
        fres = FitResults(n_sites=1, lib=MagicMock(spec=pyfdd.Lib2dl))

    def test_init_should_raises_error_for_bad_input(self):
        with self.assertRaises(TypeError):
            FitResults(n_sites='', lib=MagicMock(spec=pyfdd.Lib2dl))
        with self.assertRaises(TypeError):
            FitResults(n_sites=1, lib='')

    def test_results_list_should_increases_with_entries(self):
        more_results = {'sites': (1,2,3),
                        'cost_function': 'chi2',
                        'in_range': True}
        ft_obj_mock = MagicMock(spec=pyfdd.Fit)
        ft_obj_mock.results = dict()
        ft_obj_mock.get_dof = Mock(return_value=100)
        fres = FitResults(n_sites=1, lib=MagicMock(spec=pyfdd.Lib2dl))
        # 3 entries
        fres.append_entry(ft=ft_obj_mock, **more_results)
        fres.append_entry(ft=ft_obj_mock, **more_results)
        fres.append_entry(ft=ft_obj_mock, **more_results)

        self.assertEqual(len(fres.data_list), 3)

    def test_run_fit_and_save(self):
        lib = Lib2dl('data_files/sb600g05.2dl')
        dp_pad = DataPattern('data_files/pad_dp_2M.json')

        fm = FitManager(cost_function='chi2', n_sites=2, sub_pixels=8)
        fm.set_pattern(dp_pad, lib)

        p1 = 1
        p2 = 23
        fm.run_single_fit(p1, p2, verbose_graphics=False, get_errors=True)
        fm.run_single_fit(p1, p2, verbose_graphics=False, get_errors=True)
        fm.run_single_fit(p1, p2, verbose_graphics=False, get_errors=True)
        fm.save_output('test_save_horizontal.txt', layout='horizontal')
        os.remove('test_save_horizontal.txt')
        fm.save_output('test_save_vertical.txt', layout='vertical')
        os.remove('test_save_vertical.txt')


if __name__ == '__main__':
    unittest.main()
