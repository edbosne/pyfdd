#!/usr/bin/env python3

"""
FitManager is the user class for fitting.
"""

# Imports from standard library
import os
import warnings
import collections

# Imports from 3rd party
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import pyfdd
# Imports from project
from pyfdd.core.lib2dl import Lib2dl
from pyfdd.core.patterncreator import PatternCreator
from pyfdd.core.datapattern import DataPattern
from pyfdd.core.fit import Fit
from pyfdd.core.fitparameters import FitParameters


class FitResults:
    """
    Class for storing and exporting results from the FitManager.
    """

    def __init__(self, n_sites:int, lib:Lib2dl):

        if not isinstance(n_sites, int):
            raise TypeError(f'kw argument n_sites should be of type int, not {type(n_sites)}')
        if not isinstance(lib, Lib2dl):
            raise TypeError(f'kw argument lib should be of type Lib2dl, not {type(lib)}')

        self._n_sites = n_sites
        self._lib = lib

        self.data_list = list()

        # order of columns in results
        self.columns_horizontal = \
            ('value', 'D.O.F.', 'x', 'x_err', 'y', 'y_err', 'phi', 'phi_err',
             'counts', 'counts_err', 'sigma', 'sigma_err')
        self.columns_template = \
            ('site{:d} n', 'p{:d}', 'site{:d} description', 'site{:d} factor', 'site{:d} u1',
             'site{:d} fraction', 'fraction{:d}_err')
        for i in range(self._n_sites):
            for k in self.columns_template:
                self.columns_horizontal += (k.format(i + 1),)
        self.columns_horizontal += ('success', 'orientation gradient', 'data in sim. range')

        self.columns_vertical = \
            ('value', 'D.O.F.', 'x', 'x_err', 'y', 'y_err', 'phi', 'phi_err',
             'counts', 'counts_err', 'sigma', 'sigma_err')
        self.columns_vertical += \
            ('site n', 'p', 'site description', 'site factor', 'site u1',
             'site fraction', 'fraction_err')
        self.columns_vertical += ('success', 'orientation gradient', 'data in sim. range')

        self.df_horizontal = pd.DataFrame(data=None, columns=self.columns_horizontal)
        self.df_vertical = pd.DataFrame(data=None)  # , columns=self.columns_vertical)  # columns are set during filling

    def append_entry(self, ft:Fit, **kwarg):
        ## expected kw arguments sites, cost_function, in_range
        expected_kw = ('sites', 'cost_function', 'in_range')
        # Check if all expected keywords are present in kwarg
        for kw in expected_kw:
            if kw not in kwarg:
                raise ValueError(f"Expected keyword argument '{kw}' not found.")

        results_entry = dict()
        results_entry['parameter_dict'] = ft.get_parameters_dict()
        results_entry['ft_results'] = ft.results.copy()
        results_entry['dof'] = ft.get_dof()
        results_entry['sites'] = kwarg['sites']
        results_entry['cost_function'] = kwarg['cost_function']
        results_entry['in_range'] = kwarg['in_range'] #  self.is_datapattern_inrange(orientation_values)

        self.data_list.append(results_entry)

    def generate_results_table(self, layout='horizontal', simplify:bool = True):
        if layout not in ('horizontal', 'vertical'):
            raise ValueError(f"Argument 'layout' shoudl be either 'horizontal' or 'vertical' not {layout}.")

        # initialization
        if layout == 'horizontal':
            self.df_horizontal = pd.DataFrame(data=None, columns=self.columns_horizontal)
        elif layout == 'vertical':
            self.df_vertical = pd.DataFrame(data=None)  # , columns=self.columns_vertical)  # columns are set during filling

        # Fill dataframe
        if layout == 'horizontal':
            for entry in self.data_list:
                self._fill_horizontal_results_dict(entry)
            df =  self.df_horizontal
        elif layout == 'vertical':
            for entry in self.data_list:
                self._fill_vertical_results_dict(entry)
            df =  self.df_vertical

        if simplify:
            # Filter columns containing "_err"
            cols_to_drop = df.filter(like='_err').columns
            df = df.drop(columns=cols_to_drop)
            also_drop = ['orientation gradient', 'data in sim. range']
            df = df.drop(columns=also_drop)

        return df

    def save_as(self, filename, layout='horizontal'):
        df = self.generate_results_table(layout)

        base_name, ext = os.path.splitext(filename)
        if ext == '.txt' or ext == '.csv':
            df.to_csv(filename)
        elif ext == '.xlsx' or ext == '.xls':
            df.to_excel(filename)
        else:
            raise ValueError('Extention not recognized, use txt, csv, xls or xlsx')

    def _fill_horizontal_results_dict(self, entry:dict):#p1=None, p2=None, p3=None):

        # keys are 'pattern_1','pattern_2','pattern_3','sub_pixels','dx','dy','phi',
        # 'total_cts','sigma','f_p1','f_p2','f_p3'
        parameter_dict = entry['parameter_dict']
        ft_results = entry['ft_results']

        append_dic = dict()
        append_dic['value'] = ft_results['fun']
        append_dic['success'] = ft_results['success']
        append_dic['orientation gradient'] = np.linalg.norm(ft_results['orientation jac'])
        append_dic['data in sim. range'] = entry['in_range']
        append_dic['D.O.F.'] = entry['dof']
        append_dic['x'] = parameter_dict['dx']['value']
        append_dic['y'] = parameter_dict['dy']['value']
        append_dic['phi'] = parameter_dict['phi']['value']
        append_dic['counts'] = parameter_dict['total_cts']['value'] if entry['cost_function'] == 'chi2' else np.nan
        append_dic['sigma'] = parameter_dict['sigma']['value']

        for i in range(self._n_sites):
            patt_num = entry['sites'][i] # index of the pattern in dict_2dl is patt_num - 1
            append_dic['site{:d} n'.format(i + 1)] = self._lib.dict_2dl["Spectrums"][patt_num - 1]["Spectrum number"]
            append_dic['p{:d}'.format(i + 1)] = patt_num
            append_dic['site{:d} description'.format(i + 1)] = \
                self._lib.dict_2dl["Spectrums"][patt_num - 1]["Spectrum_description"]
            append_dic['site{:d} factor'.format(i + 1)] = self._lib.dict_2dl["Spectrums"][patt_num - 1]["factor"]
            append_dic['site{:d} u1'.format(i + 1)] = self._lib.dict_2dl["Spectrums"][patt_num - 1]["u1"]
            append_dic['site{:d} fraction'.format(i + 1)] = parameter_dict['f_p{:d}'.format(i + 1)]['value']

        # if get_errors:
        append_dic['x_err'] = parameter_dict['dx']['std']
        append_dic['y_err'] = parameter_dict['dy']['std']
        append_dic['phi_err'] = parameter_dict['phi']['std']
        append_dic['counts_err'] = parameter_dict['total_cts']['std'] if entry['cost_function'] == 'chi2' else np.nan
        append_dic['sigma_err'] = parameter_dict['sigma']['std']
        for i in range(self._n_sites):
            append_dic['fraction{:d}_err'.format(i + 1)] = \
                parameter_dict['f_p{:d}'.format(i + 1)]['std']

        self.df_horizontal = self.df_horizontal.append(append_dic, ignore_index=True)
        self.df_horizontal = self.df_horizontal[list(self.columns_horizontal)]

    def _fill_vertical_results_dict(self, entry:dict):#p1=None, p2=None, p3=None):

        # keys are 'pattern_1','pattern_2','pattern_3','sub_pixels','dx','dy','phi',
        # 'total_cts','sigma','f_p1','f_p2','f_p3'
        parameter_dict = entry['parameter_dict']
        ft_results = entry['ft_results']

        append_dic = dict()
        append_dic['value'] = ft_results['fun']
        append_dic['success'] = ft_results['success']
        append_dic['orientation gradient'] = np.linalg.norm(ft_results['orientation jac'])
        orientation_values = {'dx': parameter_dict['dx']['value'],
                              'dy': parameter_dict['dy']['value'],
                              'phi': parameter_dict['phi']['value']}
        append_dic['data in sim. range'] = entry['in_range']
        append_dic['D.O.F.'] = entry['dof']
        append_dic['x'] = parameter_dict['dx']['value']
        append_dic['y'] = parameter_dict['dy']['value']
        append_dic['phi'] = parameter_dict['phi']['value']
        append_dic['counts'] = parameter_dict['total_cts']['value'] if entry['cost_function'] == 'chi2' else np.nan
        append_dic['sigma'] = parameter_dict['sigma']['value']

        # if get_errors:
        append_dic['x_err'] = parameter_dict['dx']['std']
        append_dic['y_err'] = parameter_dict['dy']['std']
        append_dic['phi_err'] = parameter_dict['phi']['std']
        append_dic['counts_err'] = parameter_dict['total_cts']['std'] if entry['cost_function'] == 'chi2' else np.nan
        append_dic['sigma_err'] = parameter_dict['sigma']['std']

        # print('append_dic ', append_dic)
        main_columns = pd.DataFrame().append(append_dic, ignore_index=True)

        for i in range(self._n_sites):
            patt_num = entry['sites'][i]  # index of the pattern in dict_2dl is patt_num - 1
            if i == 0:
                append_dic = dict()
                append_dic['site n'] = [self._lib.dict_2dl["Spectrums"][patt_num - 1]["Spectrum number"], ]
                append_dic['p'] = [patt_num, ]
                append_dic['site description'] = \
                    [self._lib.dict_2dl["Spectrums"][patt_num - 1]["Spectrum_description"], ]
                append_dic['site factor'] = [self._lib.dict_2dl["Spectrums"][patt_num - 1]["factor"], ]
                append_dic['site u1'] = [self._lib.dict_2dl["Spectrums"][patt_num - 1]["u1"], ]
                append_dic['site fraction'] = [parameter_dict['f_p{:d}'.format(i + 1)]['value'], ]
                # if get_errors:
                append_dic['fraction_err'] = \
                    [parameter_dict['f_p{:d}'.format(i + 1)]['std'], ]
            else:
                append_dic['site n'] += [self._lib.dict_2dl["Spectrums"][patt_num - 1]["Spectrum number"], ]
                append_dic['p'] += [patt_num, ]
                append_dic['site description'] += \
                    [self._lib.dict_2dl["Spectrums"][patt_num - 1]["Spectrum_description"], ]
                append_dic['site factor'] += [self._lib.dict_2dl["Spectrums"][patt_num - 1]["factor"], ]
                append_dic['site u1'] += [self._lib.dict_2dl["Spectrums"][patt_num - 1]["u1"], ]
                append_dic['site fraction'] += [parameter_dict['f_p{:d}'.format(i + 1)]['value'], ]
                # if get_errors:
                append_dic['fraction_err'] += \
                    [parameter_dict['f_p{:d}'.format(i + 1)]['std'], ]

        temp_df = pd.concat([main_columns, pd.DataFrame.from_dict(append_dic)],
                                     axis=1, ignore_index=False)

        self.df_vertical = self.df_vertical.append(temp_df, ignore_index=True, sort=False)
        self.df_vertical = self.df_vertical[list(self.columns_vertical)]


class FitManager:
    """
    The class FitManager is a helper class for using Fit in pyfdd.
    You should be able to do all standard routine analysis from FitManager.
    It also help in creating graphs, using fit options and saving results.
    """

    default_profiles_fit_options = {
            # likelihood values are orders of mag bigger than chi2, so they need smaller ftol
            # real eps is the eps in fit options times the parameter scale
            'coarse': {'ml': {'disp': False, 'maxiter': 10, 'maxfun': 200, 'ftol': 1e-7, 'maxls': 50,
                              'maxcor': 10, 'eps': 1e-8},
                       'chi2': {'disp': False, 'maxiter': 10, 'maxfun': 200, 'ftol': 1e-6, 'maxls': 50,
                                'maxcor': 10, 'eps': 1e-8}},
            'default': {'ml': {'disp': False, 'maxiter': 20, 'maxfun': 200, 'ftol': 1e-9, 'maxls': 100,
                               'maxcor': 10, 'eps': 1e-8},  # maxfun to 200 prevents memory problems,
                        'chi2': {'disp': False, 'maxiter': 20, 'maxfun': 300, 'ftol': 1e-6, 'maxls': 100,
                                 'maxcor': 10, 'eps': 1e-8}},
            'fine': {'ml': {'disp': False, 'maxiter': 60, 'maxfun': 1200, 'ftol': 1e-12, 'maxls': 100,
                            'maxcor': 10, 'eps': 1e-8},
                     'chi2': {'disp': False, 'maxiter': 60, 'maxfun': 1200, 'ftol': 1e-9, 'maxls': 100,
                              'maxcor': 10, 'eps': 1e-8}}
    }

    # settings methods
    def __init__(self, *, cost_function='chi2', n_sites, sub_pixels=1):
        """
        FitManager is a helper class for using Fit in pyfdd.
        :param cost_function: The type of cost function to use. Possible values are 'chi2' for chi-square
        and 'ml' for maximum likelihood.
        :param sub_pixels: The number of subpixels to integrate during fit in x and y.
        """

        if cost_function not in ('chi2', 'ml'):
            raise ValueError('cost_function not valid. Use chi2 or ml')

        if not isinstance(sub_pixels, (int, np.integer)):
            raise ValueError('sub_pixels must be of type int')

        self.done_param_verbose = False

        # Output
        self.verbose = 1
        self.results = None

        # Stored objects
        self.min_value = None
        self.best_fit = None
        self.last_fit = None
        self.dp_pattern = None
        self.lib = None
        self.current_fit_obj = None

        # Fit settings
        self._n_sites = n_sites
        self._sub_pixels = sub_pixels
        self._cost_function = cost_function
        self._fit_options = {}
        self._fit_options_profile = 'default'
        self._minimization_method = 'L-BFGS-B'
        self._profiles_fit_options = FitManager.default_profiles_fit_options.copy()
        self.set_minimization_settings()
        self.fit_parameters: FitParameters = FitParameters(n_sites=self._n_sites)

    def _print(self, *msg):
        """
        This method is overwriten on the GUI to print to the message box.
        :param msg:
        :return:
        """
        print(*msg)

    def _print_settings(self, ft):
        """
        prints the settings that are in use during fit
        :param ft: Fit object
        """
        assert isinstance(ft, Fit)
        self._print('\n')
        self._print('Fit settings')
        self._print('Cost function       -', self._cost_function)
        self._print('Minimization method -', self._minimization_method)
        self._print('Fit option profile  -', self._fit_options_profile)
        self._print('Fit options         -', self._fit_options)
        self._print('Sub pixels          -', self._sub_pixels)
        self._print(self.fit_parameters)
        self._print('\n')

        self.done_param_verbose = True

    def set_pattern(self, data_pattern, library):
        """
        Set the pattern to fit.
        :param data_pattern: path or DataPattern
        :param library: path or Lib2dl
        """
        if isinstance(data_pattern, DataPattern):
            # all good
            self.dp_pattern = data_pattern
        elif isinstance(data_pattern,  str):
            if not os.path.isfile(data_pattern):
                raise ValueError('data is a str but filepath is not valid')
            else:
                self.dp_pattern = DataPattern(file_path=data_pattern, verbose=self.verbose)
        else:
            ValueError('data_pattern input error')
        self.fit_parameters.update_initial_values_with_datapattern(self.dp_pattern)
        self.fit_parameters.update_bounds_with_datapattern(self.dp_pattern)

        if isinstance(library, Lib2dl):
            # all good
            self.lib = library
        elif isinstance(library, str):
            if not os.path.isfile(library):
                raise ValueError('data is a str but filepath is not valid')
            else:
                self.lib = Lib2dl(library)
        else:
            ValueError('data_pattern input error')

        #  Reset fit results
        self.results = FitResults(self._n_sites, self.lib)

    def set_minimization_settings(self, profile='default', min_method='L-BFGS-B', options=None):
        """
        Set the options for the minimization.
        :param profile: choose between 'coarse', 'default' and 'fine', predefined options.
        :param min_method: minimization algorith to use.
        :param options: python dict with options to use. overwrites profile.
        """
        # Using a coarse profile will lead to faster results but less optimized.
        # Using a fine profile can lead to rounding errors and jumping to other minima which causes artifacts
        # scipy default eps is 1e-8 this, sometimes, is too small to correctly get the derivative of phi
        if not isinstance(options, dict) and options is not None:
            raise ValueError('options must be of type dict.')

        if options is None:
            options = dict()

        if not isinstance(min_method, str):
            raise ValueError('min_method must be of type str.')

        if profile not in ('coarse', 'default', 'fine', 'custom'):
            raise ValueError('profile value should be set to: coarse, default or fine.')

        self._minimization_method = min_method

        if len(options) > 0:
            self._fit_options = options
            self._fit_options_profile = 'custom'

        elif min_method == 'L-BFGS-B':
            self._fit_options_profile = profile
            self._fit_options = self._profiles_fit_options[profile][self._cost_function]

        else:
            warnings.warn('No profile for method {} and no options provided. Using library defaults'.format(min_method))

    def get_pattern_counts(self, ignore_masked=True):
        """
        Get the total counts of the pattern to be fit
        :param ignore_masked:
        :return:
        """

        if not isinstance(ignore_masked, bool):
            raise ValueError('ignore_masked must be of type bool.')

        if self.dp_pattern is None:
            raise ValueError('The data_pattern is not properly set.')

        if ignore_masked:
            total_cts = self.dp_pattern.pattern_matrix.sum()
        else:
            total_cts = self.dp_pattern.pattern_matrix.data.sum()
        return total_cts

    def _pass_resuts_to_initial_values(self, pass_results=False):
        """
        Get the initial values for the next fit
        :param pass_results: Use the previous fit results
        :return p0, p_fix: initial values and tuple of bools indicating if it is fixed
        """
        # decide if using last fit results
        p0_pass = pass_results \
            and self.last_fit is not None \
            and self.last_fit.results['success']

        keys = self.fit_parameters.get_keys()
        initial_values_dict = self.fit_parameters.get_initial_values()
        fixed_values_dict = self.fit_parameters.get_fixed_values()

        if p0_pass:  # change initial values
            p0_last = self.last_fit.results['x']

            for key, last_value in zip(keys, p0_last):
                if fixed_values_dict[key]:
                    pass  # Fixed, dont change
                else:
                    # starting too close from a minimum can cause errors so 1e-5 is added
                    initial_values_dict[key] = last_value + 1e-5

        return initial_values_dict

    def _build_fits_obj(self, sites, verbose_graphics=False, pass_results=False):
        """
        Builds a Fit object
        :param sites: list of sites to fit
        :param verbose_graphics: plot pattern as it is being fit
        :param pass_results: argument for _get_initial_values
        :return: Fit object
        """

        ft = Fit(self.lib, sites, verbose_graphics)

        ft.set_sub_pixels(self._sub_pixels)
        ft.set_fit_options(self._fit_options)

        patt = self.dp_pattern.pattern_matrix
        xmesh = self.dp_pattern.xmesh
        ymesh = self.dp_pattern.ymesh
        ft.set_data_pattern(xmesh, ymesh, patt)

        # Get parameter values
        init_dict = self.fit_parameters.get_initial_values()
        fix_dict = self.fit_parameters.get_fixed_values()
        scale_dict = self.fit_parameters.get_step_modifier()
        bound_dict = self.fit_parameters.get_bounds()

        if pass_results:  # Overwrite previous init dict
            init_dict = self._pass_resuts_to_initial_values()

        ft.set_scale_values(**scale_dict)
        ft.set_initial_values(**init_dict)
        ft.fix_parameters(**fix_dict)
        ft.set_bound_values(**bound_dict)

        return ft

    def run_fits(self, *args, pass_results=False, verbose=1, get_errors=False):
        """
        Run Fit for a list of sites.
        :param args: list of patterns for each site. Up to tree sites are possible
        :param pass_results: Use the last fit parameter results as input for the next.
        :param verbose: 0 silent, 1 default and 2 max verbose
        :param get_errors: Calculate hessian matrix to get error values after fit
        :return:
        """

        if not self.is_datapattern_inrange():
            warnings.warn('The datapattern is not in the simulation range. \n'
                          'Consider reducing the fit range arount the axis.')
            raise ValueError

        self.done_param_verbose = False

        patterns_list = ()
        #print('args, ',args)
        for ar in args:
            # if a pattern index is just a scalar make it iterable
            patterns_list += (np.atleast_1d(np.array(ar)),)
        assert len(patterns_list) >= 1

        print(f'patterns_list {patterns_list}')

        def recursive_call(patterns_list, sites = ()):
            #print('patterns_list, sites -', patterns_list, sites)
            if len(patterns_list) > 0:
                for s in patterns_list[0]:
                    sites_new = sites + (s,)
                    recursive_call(patterns_list[1:], sites_new)
            else:
                # visualization is by default off in run_fits
                self._single_fit(sites, verbose=verbose, pass_results=pass_results, get_errors=get_errors)

        try:
            recursive_call(patterns_list)
        except:
            # Reset current fit object
            print('Recursive call did not work')
            self.current_fit_obj = None

    def run_single_fit(self, *args, verbose=1,
                       verbose_graphics=False, get_errors=False):

        if not self.is_datapattern_inrange():
            warnings.warn('The datapattern is not in the simulation range. \n'
                          'Consider reducing the fit range arount the axis.')

        args = list(args)
        sites = ()
        for i in range(len(args)):
            # Convert array of single number to scalar
            if isinstance(args[i], (np.ndarray, collections.abc.Sequence)) and len(args[i]) == 1:
                args[i] = args[i][0]
            # Ensure index is an int.
            if not isinstance(args[i], (int, np.integer)):
                raise ValueError('Each pattern index must an int.')
            sites += (args[i],)

        self.done_param_verbose = False

        try:
            self._single_fit(sites, get_errors=get_errors, pass_results=False,
                         verbose=verbose, verbose_graphics=verbose_graphics)
        except:
            # Reset current fit object
            self.current_fit_obj = None

    def _single_fit(self, sites, get_errors=False, pass_results=False,
                    verbose=1, verbose_graphics=False):

        if not isinstance(sites, collections.abc.Sequence):
            if isinstance(sites, (int, np.integer)):
                sites = (sites,)
            else:
                raise ValueError('sites needs to be an int or a sequence of ints')
        for s in sites:
            if not isinstance(s, (int, np.integer)):
                raise ValueError('sites needs to be an int or a sequence of ints')

        # Ensure the number of sites indexes is the same as the number of sites in __init__
        if len(sites) != self._n_sites:
            raise ValueError('Error, you need to input the pattern indices for all the '
                             '{0} expected sites. {1} were provided. '
                             'The expected number of sites can be '
                             'changed in the constructor.'.format(self._n_sites, len(sites)))

        # sanity check
        assert isinstance(verbose_graphics, bool)
        assert isinstance(get_errors, bool)
        assert isinstance(self.dp_pattern, DataPattern)

        self.current_fit_obj = self._build_fits_obj(sites, verbose_graphics, pass_results=pass_results)

        if verbose > 0 and self.done_param_verbose is False:
            self._print_settings(self.current_fit_obj)

        if verbose > 0:
            self._print('Sites (P1, P2, ...) - ', sites)

        self.current_fit_obj.minimize_cost_function(self._cost_function)

        if verbose > 1:
            print(self.current_fit_obj.results)

        if get_errors:
            self.current_fit_obj.get_std_from_hessian(self.current_fit_obj.results['x'], enable_scale=True, func=self._cost_function)

        more_results = {'sites': sites,
                        'cost_function': self._cost_function,
                        'in_range': self.is_datapattern_inrange(self.current_fit_obj)}

        self.results.append_entry(self.current_fit_obj, **more_results)


        # Keep best fit
        if self.min_value is None:
            self.best_fit = self.current_fit_obj
            self.min_value = self.current_fit_obj.results['fun']
        elif self.current_fit_obj.results['fun'] < self.min_value:
           self.best_fit = self.current_fit_obj
           self.min_value = self.current_fit_obj.results['fun']

        self.last_fit = self.current_fit_obj

        # Reset current fit object
        self.current_fit_obj = None

    def stop_current_fit(self):
        if self.current_fit_obj is not None:
            self.current_fit_obj.stop_current_fit()

    def is_datapattern_inrange(self, fit_obj=None):

        if isinstance(fit_obj, Fit):
            parameter_dict = fit_obj.get_parameters_dict()
            orientation_values = {'dx': parameter_dict['dx']['value'],
                                  'dy': parameter_dict['dy']['value'],
                                  'phi': parameter_dict['phi']['value']}

            if orientation_values is not None and len(orientation_values) < 3:
                raise ValueError('Orientation_values need to be at least of lenght 3.')

        elif fit_obj is None:
            orientation_values = self.fit_parameters.get_initial_values()
        else:
            raise TypeError(f'argument fit_obj should be of type pyfdd.Fit or None, not {type(fit_obj)}.')

        dx = orientation_values['dx']
        dy = orientation_values['dy']
        phi = orientation_values['phi']

        # generate sim pattern
        gen = PatternCreator(self.lib, self.dp_pattern.xmesh, self.dp_pattern.ymesh, 1,
                             mask=self.dp_pattern.pattern_matrix.mask,  # need the mask for the normalization
                             sub_pixels=self._sub_pixels,
                             mask_out_of_range=True)
        # mask out of range false means that points that are out of the range of simulations are not masked,
        # instead they are substituted by a very small number 1e-12
        sim_pattern = gen.make_pattern(dx, dy, phi, 0, 1, sigma=0, pattern_type='ideal')

        # Logic verification
        # Data points that are not masked should not be in a position where the simulation is masked
        data_mask = self.dp_pattern.pattern_matrix.mask
        sim_mask = sim_pattern.mask
        inrange = not np.any(~data_mask == sim_mask)

        return inrange

    # results and output methods
    def save_output(self, filename, layout='horizontal', save_figure=False):

        if self.results is None:
            return

        assert isinstance(self.results, FitResults)

        self.results.save_as(filename, layout)

        if save_figure:
            base_name, ext = os.path.splitext(filename)
            xmesh = self.best_fit.XXmesh
            ymesh = self.best_fit.YYmesh
            # data pattern
            fig = plt.figure()
            plt.contourf(xmesh, ymesh, self.best_fit.data_pattern)
            plt.colorbar()
            fig.savefig(base_name + '_data.png')
            plt.close(fig)
            # sim pattern
            fig = plt.figure()
            plt.contourf(xmesh, ymesh, self.best_fit.sim_pattern)
            plt.colorbar()
            fig.savefig(base_name + '_sim.png')
            plt.close(fig)
            # sim-data pattern
            fig = plt.figure()
            plt.contourf(xmesh, ymesh, self.best_fit.sim_pattern - self.best_fit.data_pattern)
            plt.colorbar()
            fig.savefig(base_name + '_sim-data.png')
            plt.close(fig)

    def _get_sim_normalization_factor(self, normalization, pattern_type, fit_obj=None):

        assert isinstance(fit_obj, Fit) or fit_obj is None
        total_counts = np.sum(self.dp_pattern.pattern_matrix)
        if fit_obj is None:
            total_yield = None
        else:
            sim_pattern = self._gen_detector_pattern_from_fit(fit_obj=fit_obj, generator='yield', rm_mask=False)
            total_yield = sim_pattern.sum()
            # print('total_yield', total_yield, '# pixels', np.sum(~sim_pattern.mask))
            #total_yield = np.sum(~self.dp_pattern.pattern_matrix.mask)
        norm_factor = None
        if normalization is None:
            norm_factor = 1
        elif normalization == 'counts':
            if pattern_type == 'chi2' or pattern_type == 'data':
                norm_factor = 1
            elif pattern_type == 'ml':
                norm_factor = total_counts
        elif normalization == 'yield':
            if total_yield is None:
                raise ValueError('Simulation pattern is not defined.')
            if pattern_type == 'chi2' or pattern_type == 'data':
                norm_factor = total_yield / total_counts
            elif pattern_type == 'ml':
                norm_factor = total_yield
        elif normalization == 'probability':
            if  pattern_type == 'chi2' or pattern_type == 'data':
                norm_factor = 1 / total_counts
            elif pattern_type == 'ml':
                norm_factor = 1
        else:
            raise ValueError('normalization needs to be, None, \'counts\', \'yield\' or \'probability\'')
        return norm_factor

    def _gen_detector_pattern_from_fit(self, fit_obj, generator='ideal', rm_mask=False):

        assert isinstance(fit_obj, Fit)

        # get values
        parameter_dict = fit_obj._parameters_dict.copy()
        dx = parameter_dict['dx']['value']
        dy = parameter_dict['dy']['value']
        phi = parameter_dict['phi']['value']
        total_events = parameter_dict['total_cts']['value'] if self._cost_function == 'chi2' else \
            np.sum(self.dp_pattern.pattern_matrix)
        sigma = parameter_dict['sigma']['value']
        fractions_sims = ()
        for i in range(self._n_sites):
            fractions_sims += (parameter_dict['f_p{:d}'.format(i + 1)]['value'],)

        # generate sim pattern
        gen = PatternCreator(fit_obj._lib, fit_obj.XXmesh, fit_obj.YYmesh, fit_obj._sites_idx,
                             mask=fit_obj.data_pattern.mask, # need the mask for the normalization
                             sub_pixels=parameter_dict['sub_pixels']['value'],
                             mask_out_of_range = True)
        # mask out of range false means that points that are out of the range of simulations are not masked,
        # instead they are substituted by a very small number 1e-12
        sim_pattern = gen.make_pattern(dx, dy, phi, fractions_sims, total_events, sigma=sigma, pattern_type=generator)

        # Substitute only masked pixels that are in range (2.7° from center) and are not the chip edges
        # This can't really be made without keeping 2 set of masks, so all masked pixels are susbstituted.
        # This means some pixels with valid data but masked can still be susbtituted
        if rm_mask:
            # only mask what is outside of simulation range
            sim_pattern_ideal = gen.make_pattern(dx, dy, phi, fractions_sims, total_events, sigma=sigma, pattern_type='ideal')
            return ma.array(sim_pattern.data, mask=(sim_pattern_ideal.data == 0))
        else:
            return sim_pattern

    def get_pattern_from_last_fit(self, normalization=None):
        fit_obj = self.last_fit
        assert isinstance(fit_obj, Fit)
        #print(fit_obj.sim_pattern.data)

        norm_factor = \
            self._get_sim_normalization_factor(normalization, pattern_type=self._cost_function, fit_obj=fit_obj)

        dp = DataPattern(pattern_array=fit_obj.sim_pattern.data)
        dp.set_xymesh(fit_obj.XXmesh, fit_obj.YYmesh)
        dp.set_mask(fit_obj.sim_pattern.mask)

        return dp * norm_factor

    def get_pattern_from_best_fit(self, normalization=None):
        fit_obj = self.best_fit
        assert isinstance(fit_obj, Fit)
        #print(fit_obj.sim_pattern.data)

        norm_factor = \
            self._get_sim_normalization_factor(normalization, pattern_type=self._cost_function, fit_obj=fit_obj)

        dp = DataPattern(pattern_array=fit_obj.sim_pattern.data)
        dp.set_xymesh(fit_obj.XXmesh, fit_obj.YYmesh)
        dp.set_mask(fit_obj.sim_pattern.mask)
        return dp * norm_factor

    def get_datapattern(self, normalization=None, substitute_masked_with=None, which_fit='last'):

        # which_fit can be the best or last
        if which_fit == 'best':
            fit_obj = self.best_fit
        elif which_fit == 'last':
            fit_obj = self.last_fit
        else:
            raise ValueError('parameter fit must be either \'best\' or \'last\'')

        dp_pattern = self.dp_pattern.copy()

        if substitute_masked_with is not None:
            # Get a pattern with no mask besides what is outside of the simulation.
            sim_pattern = self._gen_detector_pattern_from_fit(fit_obj=fit_obj, generator=substitute_masked_with,
                                                              rm_mask=True)

            # Substitute pixels that are masked and that are not in the fitregion mask
            substitute_matrix = np.logical_and(dp_pattern.pixels_mask,
                                               np.logical_not(sim_pattern.mask))

            dp_pattern.pattern_matrix.data[substitute_matrix] = \
                sim_pattern.data[substitute_matrix]
            dp_pattern.clear_mask(pixels_mask=True, fitregion_mask=True)

        norm_factor = self._get_sim_normalization_factor(normalization, pattern_type='data', fit_obj=fit_obj)

        return dp_pattern * norm_factor
