#!/usr/bin/env python3

'''
Fit manager is the kernel class for fitting.
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

from .lib2dl import Lib2dl
from .patterncreator import PatternCreator, create_detector_mesh
from .datapattern import DataPattern
from .fit import Fit

import pandas as pd
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import collections
import copy


class FitManager:
    '''
    The class FitManager is a helper class for using Fit in pyfdd.
    You should be able to do all standard routine analysis from FitManager.
    It also help in creating graphs, using fit options and saving results.
    '''

    # settings methods
    def __init__(self,*, cost_function='chi2', n_sites, sub_pixels=1):
        '''
        FitManager is a helper class for using Fit in pyfdd.
        :param cost_function: The type of cost function to use. Possible values are 'chi2' for chi-square
        and 'ml' for maximum likelihood.
        :param sub_pixels: The number of subpixels to integrate during fit in x and y.
        '''

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

        # Fit settings
        self._n_sites = n_sites
        self.parameter_keys = ('dx', 'dy', 'phi', 'total_cts', 'sigma')
        for i in range(self._n_sites):
            fraction_key = 'f_p' + str(i+1)  # 'f_p1', 'f_p2', 'f_p3',...
            self.parameter_keys += (fraction_key,)
        self._cost_function = cost_function
        self._fit_options = {}
        self._fit_options_profile = 'default'
        self._minimization_method = 'L-BFGS-B'
        self.set_minimization_settings()
        self._sub_pixels = sub_pixels
        # total_cts is overwriten with values from the data pattern
        self._scale = {'dx':.01, 'dy':.01, 'phi':0.10, 'total_cts':0.01,
                       'sigma':.001}
        self._bounds = {'dx': (-3, +3), 'dy': (-3, +3), 'phi': (None, None), 'total_cts': (1, None),
                         'sigma': (0.01, None)}
        scale_temp = {}
        bounds_temp = {}
        for i in range(self._n_sites):
            fraction_key = 'f_p' + str(i+1)  # 'f_p1', 'f_p2', 'f_p3',...
            # scale
            # 'f_p1':.01, 'f_p2':.01, 'f_p3':.01}
            scale_temp[fraction_key] = 0.01
            bounds_temp[fraction_key] = (0, 1)
        self._scale = {**self._scale, **scale_temp}
        self._bounds = {**self._bounds, **bounds_temp}


        # Fit parameters settings
        # overwrite defaults from Fit
        self.p_initial_values = {}
        self.p_fixed_values = {}

        # order of columns in results
        self.columns = \
            ('value', 'D.O.F.', 'x', 'x_err', 'y', 'y_err', 'phi', 'phi_err',
             'counts', 'counts_err', 'sigma', 'sigma_err')
        self.columns_template = \
            ('site{:d} n', 'p{:d}', 'site{:d} description', 'site{:d} factor', 'site{:d} u1',
             'site{:d} fraction', 'fraction{:d}_err')
        for i in range(self._n_sites):
             for k in self.columns_template:
                 self.columns += (k.format(i+1),)
        self.columns += ('success',)

        self.df = pd.DataFrame(data=None, columns=self.columns)

    def set_pattern(self, data_pattern, library):
        '''
        Set the pattern to fit.
        :param data_pattern: path or DataPattern
        :param library: path or Lib2dl
        '''
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

        print('\nMedipix pattern added')
        print('Inicial orientation (x, y, phi) is (',
              self.dp_pattern.center[0], ', ', self.dp_pattern.center[1], ',',
              self.dp_pattern.angle, ')')

    def _print_settings(self, ft):
        '''
        prints the settings that are in use during fit
        :param ft: Fit object
        '''
        assert isinstance(ft, Fit)
        print('\n')
        print('Fit settings')
        print('Cost function       -', self._cost_function)
        print('Minimization method -', self._minimization_method)
        print('Fit option profile  -', self._fit_options_profile)
        print('Fit options         -', self._fit_options)
        print('Sub pixels          -', self._sub_pixels)
        print('\nParameter settings')
        print('{:<16}{:<16}{:<16}{:<16}{:<16}'.format('Name', 'Inicial Value', 'Fixed', 'Bounds', 'Scale'))
        string_temp = '{:<16}{:<16.2f}{:<16}{:<16}{:<16}'
        for key in self.parameter_keys:
            # {'p0':None, 'value':None, 'use':False, 'std':None, 'scale':1, 'bounds':(None,None)}
            print(string_temp.format(
                key,
                ft._parameters_dict[key]['p0'],
                ft._parameters_dict[key]['use'] == False,
                '({},{})'.format(ft._parameters_dict[key]['bounds'][0],ft._parameters_dict[key]['bounds'][1]),
                ft._parameters_dict[key]['scale']
            ))
        print('\n')

        self.done_param_verbose = True

    def set_initial_values(self, **kwargs):
        '''
        Set the initial values for a parameter. It might be overwriten if pass_results option is used
        :param kwargs: possible arguments are 'dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3'
        '''
        #('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        for key in kwargs.keys():
            if key in self.parameter_keys:
                self.p_initial_values[key] = kwargs[key]
            else:
                raise(ValueError, 'key word ' + key + 'is not recognized!' +
                      '\n Valid keys are, \'dx\',\'dy\',\'phi\',\'total_cts\',\'sigma\',\'f_p1\',\'f_p2\',\'f_p3\'')

    def set_fixed_values(self, **kwargs):
        '''
        Fix a parameter to a value. Overwrites initial value
        :param kwargs: possible arguments are 'dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3'
        '''
        #('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        for key in kwargs.keys():
            if key in self.parameter_keys:
                self.p_fixed_values[key] = kwargs[key]
            else:
                raise (ValueError, 'key word ' + key + 'is not recognized!' +
                       '\n Valid keys are, \'dx\',\'dy\',\'phi\',\'total_cts\',\'sigma\',\'f_p1\',\'f_p2\',\'f_p3\'')

    def set_bounds(self, **kwargs):
        '''
        Set bounds to a paramater. Bounds are a tuple with two values, for example, (0, None).
        :param kwargs: possible arguments are 'dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3'
        '''
        #('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        for key in kwargs.keys():
            if key in self.parameter_keys:
                if not isinstance(kwargs[key], tuple) or len(kwargs[key]) != 2:
                    raise (ValueError, 'Bounds must be a tuple of length 2.')
                self._bounds[key] = kwargs[key]
            else:
                raise (ValueError, 'key word ' + key + 'is not recognized!' +
                       '\n Valid keys are, \'dx\',\'dy\',\'phi\',\'total_cts\',\'sigma\',\'f_p1\',\'f_p2\',\'f_p3\'')

    def set_step_modifier(self, **kwargs):
        '''
        Set a step modifier value for a parameter.
        If a modifier of 10 is used for parameter P the fit will try step 10x the default step.
        For the L-BFGS-B minimization method the default steps are 1 for each value exept for the total counts
        that is the order of magnitude of the counts in the data pattern
        :param kwargs: possible arguments are 'dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3'
        '''
        #('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        for key in kwargs.keys():
            if key in self.parameter_keys:
                self._scale[key] = kwargs[key]
            else:
                raise (ValueError, 'key word ' + key + 'is not recognized!' +
                       '\n Valid keys are, \'dx\',\'dy\',\'phi\',\'total_cts\',\'sigma\',\'f_p1\',\'f_p2\',\'f_p3\'')

    def set_minimization_settings(self, profile='default', min_method='L-BFGS-B', options={}):
        '''
        Set the options for the minimization.
        :param profile: choose between 'coarse', 'default' and 'fine', predefined options.
        :param min_method: minimization algorith to use.
        :param options: python dict with options to use. overwrites profile.
        '''
        # Using a coarse profile will lead to faster results but less optimized.
        # Using a fine profile can lead to rounding errors and jumping to other minima which causes artifacts
        # scipy default eps is 1e-8 this, sometimes, is too small to correctly get the derivative of phi
        if not isinstance(options, dict):
            raise ValueError('options must be of type dict.')

        if not isinstance(min_method, str):
            raise ValueError('min_method must be of type str.')

        self._minimization_method = min_method

        if len(options) > 0:
            self._fit_options = options
            self._fit_options_profile = 'user'

        elif min_method == 'L-BFGS-B':
            ml_fit_options = {}
            chi2_fit_options = {}

            if profile == 'coarse':
                # likelihood values are orders of mag bigger than chi2, so they need smaller ftol
                ml_fit_options =   {'disp':False, 'maxiter':10, 'maxfun':200, 'ftol':1e-7, 'maxcor':100, 'eps':1e-5}
                chi2_fit_options = {'disp':False, 'maxiter':10, 'maxfun':200, 'ftol':1e-6, 'maxcor':100, 'eps':1e-5}
            elif profile == 'default':
                ml_fit_options =   {'disp':False, 'maxiter':20, 'maxfun':200, 'ftol':1e-9, 'maxcor':100, 'eps':1e-5} #maxfun to 200 prevents memory problems
                chi2_fit_options = {'disp':False, 'maxiter':20, 'maxfun':300, 'ftol':1e-6, 'maxcor':100, 'eps':1e-5}
            elif profile == 'fine':
                # use default eps with fine
                ml_fit_options =   {'disp':False, 'maxiter':30, 'maxfun':600, 'ftol':1e-12, 'maxcor':100, 'eps':1e-5}
                chi2_fit_options = {'disp':False, 'maxiter':30, 'maxfun':600, 'ftol':1e-7,  'maxcor':100, 'eps':1e-5}
            else:
                raise ValueError('profile value should be set to: coarse, default or fine.')

            self._fit_options_profile = profile

            if self._cost_function == 'chi2':
                self._fit_options = chi2_fit_options
            elif self._cost_function == 'ml':
                self._fit_options = ml_fit_options

        else:
            warnings.warn('No profile for method {} and no options provided. Using library defaults'.format(min_method))

    def _get_initial_values(self, pass_results=False):
        '''
        Get the initial values for the next fit
        :param pass_results: Use the previous fit results
        :return p0, p_fix: initial values and tuple of bools indicating if it is fixed
        '''
        #('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        p0 = ()
        p_fix = ()
        # decide if using last fit results
        p0_pass = pass_results \
                  and self.last_fit is not None \
                  and self.last_fit.results['success']
        # starting too close from a minimum can cause errors so 1e-5 is added
        p0_last = self.last_fit.results['x'] + 1e-5 if p0_pass else None
        #print('p0_last', p0_last)
        p0_last_i = 0
        for key in self.parameter_keys:
            # Use user defined fixed value
            if key in self.p_fixed_values:
                p0 += (self.p_fixed_values[key],)
                p_fix += (True,)
            # Use user defined initial value
            elif key in self.p_initial_values:
                p0 += (self.p_initial_values[key],)
                p_fix += (False,)
            # Use FitManager choice
            else:
                if key == 'dx':
                    p0 += (p0_last[p0_last_i],) if p0_pass else (self.dp_pattern.center[0],)
                elif key == 'dy':
                    p0 += (p0_last[p0_last_i],) if p0_pass else (self.dp_pattern.center[1],)
                elif key == 'phi':
                    p0 += (p0_last[p0_last_i],) if p0_pass else (self.dp_pattern.angle,)
                elif key == 'total_cts':
                    patt = self.dp_pattern.matrixOriginal.copy()
                    #counts_ordofmag = 10 ** (int(math.log10(patt.sum())))
                    counts= patt.sum()
                    p0 += (counts,)
                elif key == 'sigma':
                    p0 += (p0_last[p0_last_i],) if p0_pass else (0.1,)
                else:
                    # assuming a pattern fraction
                    p0 += (p0_last[p0_last_i],) if p0_pass and p0_last_i<len(p0_last) else (0.15,)
                p_fix += (False,)
                if p0_pass:
                    p0_last_i += 1
        #print('p0',p0,'\np_fix', p_fix)
        return p0, p_fix

    def _get_scale_values(self):
        scale = ()
        for key in self.parameter_keys:
            # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
            # total_cts is a spacial case at it uses the counts from the pattern
            if key == 'total_cts':
                if self._cost_function == 'chi2':
                    patt = self.dp_pattern.matrixCurrent
                    counts_ordofmag = 10 ** (int(math.log10(patt.sum())))
                    scale += (counts_ordofmag * self._scale[key],)
                elif self._cost_function == 'ml':
                    scale += (-1,)
            else:
                scale += (self._scale[key],)
        return scale

    def _build_fits_obj(self, sites, verbose_graphics=False, pass_results=False):
        '''
        Builds a Fit object
        :param p1: pattern 1
        :param p2: pattern 2
        :param p3: pattern 3
        :param verbose_graphics: plot pattern as it is being fit
        :param pass_results: argument for _get_initial_values
        :return: Fit object
        '''

        ft = Fit(self.lib, sites, verbose_graphics)

        ft.set_sub_pixels(self._sub_pixels)
        ft.set_fit_options(self._fit_options)

        patt = self.dp_pattern.matrixCurrent
        xmesh = self.dp_pattern.xmesh
        ymesh = self.dp_pattern.ymesh
        ft.set_data_pattern(xmesh, ymesh, patt)

        # Get initial values
        p0, p0_fix = self._get_initial_values(pass_results=pass_results)

        # Get scale
        scale = self._get_scale_values()

        # base input
        scale_dict = {
            'dx':scale[0], 'dy':scale[1], 'phi':scale[2], 'total_cts':scale[3], 'sigma':scale[4]
        }
        init_dict = {
            'dx':p0[0], 'dy':p0[1], 'phi':p0[2], 'total_cts':p0[3], 'sigma':p0[4]
        }
        fix_dict = {
            'dx': p0_fix[0], 'dy': p0_fix[1], 'phi': p0_fix[2], 'total_cts': p0_fix[3], 'sigma': p0_fix[4]
        }
        bound_dict = {
            'dx':self._bounds['dx'], 'dy':self._bounds['dy'], 'phi':self._bounds['phi'],
            'total_cts':self._bounds['total_cts'], 'sigma':self._bounds['sigma']
        }
        # add site values

        for i in range(self._n_sites):
            fraction_key = 'f_p' + str(i + 1)  # 'f_p1', 'f_p2', 'f_p3',...
            scale_dict[fraction_key] = scale[5+i]
            init_dict[fraction_key] = p0[5+i]
            fix_dict[fraction_key] = p0_fix[5+i]
            bound_dict[fraction_key] = self._bounds[fraction_key]

        ft.set_scale_values(**scale_dict)
        ft.set_inicial_values(**init_dict)
        ft.fix_parameters(**fix_dict)
        ft.set_bound_values(**bound_dict)

        return ft

    def _fill_results_dict(self, ft, get_errors, sites):#p1=None, p2=None, p3=None):
        assert isinstance(ft, Fit), "ft is not of type PyFDD.Fit."

        patt = self.dp_pattern.matrixCurrent.copy()

        # keys are 'pattern_1','pattern_2','pattern_3','sub_pixels','dx','dy','phi',
        # 'total_cts','sigma','f_p1','f_p2','f_p3'
        parameter_dict = ft._parameters_dict.copy()
        append_dic = {}
        append_dic['value'] = ft.results['fun']
        append_dic['success'] = ft.results['success']
        append_dic['D.O.F.'] = ft.get_dof()
        append_dic['x'] = parameter_dict['dx']['value']
        append_dic['y'] = parameter_dict['dy']['value']
        append_dic['phi'] = parameter_dict['phi']['value']
        append_dic['counts'] = parameter_dict['total_cts']['value'] if self._cost_function == 'chi2' else np.nan
        append_dic['sigma'] = parameter_dict['sigma']['value']

        for i in range(self._n_sites):
            patt_num = sites[i] # index of the pattern in ECdict is patt_num - 1
            append_dic['site{:d} n'.format(i + 1)] = self.lib.ECdict["Spectrums"][patt_num - 1]["Spectrum number"]
            append_dic['p{:d}'.format(i + 1)] = patt_num
            append_dic['site{:d} description'.format(i + 1)] = \
                self.lib.ECdict["Spectrums"][patt_num - 1]["Spectrum_description"]
            append_dic['site{:d} factor'.format(i + 1)] = self.lib.ECdict["Spectrums"][patt_num - 1]["factor"]
            append_dic['site{:d} u1'.format(i + 1)] = self.lib.ECdict["Spectrums"][patt_num - 1]["u1"]
            append_dic['site{:d} fraction'.format(i + 1)] = parameter_dict['f_p{:d}'.format(i + 1)]['value']

        if get_errors:
            append_dic['x_err'] = parameter_dict['dx']['std']
            append_dic['y_err'] = parameter_dict['dy']['std']
            append_dic['phi_err'] = parameter_dict['phi']['std']
            append_dic['counts_err'] = parameter_dict['total_cts']['std'] if self._cost_function == 'chi2' else np.nan
            append_dic['sigma_err'] = parameter_dict['sigma']['std']
            for i in range(self._n_sites):
                append_dic['fraction{:d}_err'.format(i + 1)] = \
                    parameter_dict['f_p{:d}'.format(i + 1)]['std']

        # print('append_dic ', append_dic)
        self.df = self.df.append(append_dic, ignore_index=True)
        #print('columns - ', list(self.df))
        self.df = self.df[list(self.columns)]
        # print('self.df ', self.df)

    def run_fits(self, *args, pass_results=False, verbose=1, get_errors=False):
        '''
        Run Fit for a list of sites.
        :param args: list of patterns for each site. Up to tree sites are possible
        :param pass_results: Use the last fit parameter results as input for the next.
        :param verbose: 0 silent, 1 default and 2 max verbose
        :return:
        '''

        if len(args) != self._n_sites:
            raise ValueError('Error, you need to imput the pattern idexes for all the '
                             '{0} expected sites.'.format(self._n_sites, pattern_index))

        self.done_param_verbose = False

        patterns_list = ()
        #print('args, ',args)
        for ar in args:
            # if a pattern index is just a scalar make it iterable
            patterns_list += (np.atleast_1d(np.array(ar)),)
        assert len(patterns_list) >= 1

        def recursive_call(patterns_list, sites = ()):
            #print('patterns_list, sites -', patterns_list, sites)
            if len(patterns_list) > 0:
                for s in patterns_list[0]:
                    sites_new = sites + (s,)
                    recursive_call(patterns_list[1:], sites_new)
            else:
                # visualization is by default off in run_fits
                self._single_fit(sites, verbose=verbose, pass_results=pass_results, get_errors=get_errors)

        recursive_call(patterns_list)

    def run_single_fit(self, *args, verbose=1,
                       verbose_graphics=False, get_errors=False):

        args = list(args)
        sites = ()
        for i in range(len(args)):
            # Convert array of single number to scalar
            if isinstance(args[i], (np.ndarray, collections.Sequence)) and len(args[i]) == 1:
                args[i] = args[i][0]
            # Ensure index is an int.
            if not isinstance(args[i], (int, np.integer)):
                raise ValueError('Each pattern index must an int.')
            sites += (args[i],)

        self.done_param_verbose = False

        self._single_fit(sites, get_errors=get_errors, pass_results=False,
                         verbose=verbose, verbose_graphics=verbose_graphics)

    def _single_fit(self, sites, get_errors=False, pass_results=False,
                    verbose=1, verbose_graphics=False):

        if not isinstance(sites, collections.Sequence):
            if isinstance(sites, (int, np.integer)):
                sites = (sites,)
            else:
                raise ValueError('sites needs to be an int or a sequence of ints')
        for s in sites:
            if not isinstance(s, (int, np.integer)):
                raise ValueError('sites needs to be an int or a sequence of ints')

        # Ensure the number of sites indexes is the same as the number of sites in __init__
        if len(sites) != self._n_sites:
            raise ValueError('Error, you need to imput the pattern idices for all the '
                             '{0} expected sites. {1} were provided. '
                             'The expected number of sites can be '
                             'changed in the constructor.'.format(self._n_sites, len(sites)))

        # sanity check
        assert isinstance(verbose_graphics, bool)
        assert isinstance(get_errors, bool)
        assert isinstance(self.dp_pattern, DataPattern)

        ft = self._build_fits_obj(sites, verbose_graphics, pass_results=pass_results)

        if verbose > 0 and self.done_param_verbose is False:
            self._print_settings(ft)

        if verbose > 0:
            print('Sites (P1, P2, ...) - ', sites)

        ft.minimize_cost_function(self._cost_function)

        if verbose > 1:
            print(ft.results)

        if get_errors:
            ft.get_std_from_hessian(ft.results['x'], func=self._cost_function)

        self._fill_results_dict(ft, get_errors, sites)

        # Keep best fit
        if self.min_value is None:
            self.best_fit = ft
            self.min_value = ft.results['fun']
        elif ft.results['fun'] < self.min_value:
           self.best_fit = ft
           self.min_value = ft.results['fun']

        self.last_fit = ft

    # results and output methods
    def save_output(self, filename, save_figure=False):
        self.df.to_csv(filename)
        base_name, ext = os.path.splitext(filename)
        if ext == '.txt' or ext == '.csv':
            self.df.to_csv(filename)
        elif ext == '.xlsx' or ext == '.xls':
            self.df.to_csv(filename)
        else:
            raise ValueError('Extention not recognized, use txt, csv, xls or xlsx')

        if save_figure:
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

    def _get_sim_normalization_factor(self, normalization, pattern_type):
        total_counts = np.sum(self.dp_pattern.matrixCurrent)
        num_pix = np.sum(~self.dp_pattern.matrixCurrent.mask)
        norm_factor = None
        if normalization is None:
            norm_factor = 1
        elif normalization == 'counts':
            if pattern_type == 'chi2' or pattern_type == 'data':
                norm_factor = 1
            elif pattern_type == 'ml':
                norm_factor = total_counts
        elif normalization == 'yield':
            if pattern_type == 'chi2' or pattern_type == 'data':
                norm_factor = num_pix / total_counts
            elif pattern_type == 'ml':
                norm_factor = num_pix
        elif normalization == 'probability':
            if  pattern_type == 'chi2' or pattern_type == 'data':
                norm_factor = 1 / total_counts
            elif pattern_type == 'ml':
                norm_factor = 1
        else:
            raise ValueError('normalization needs to be, None, \'counts\', \'yield\' or \'probability\'')
        return norm_factor

    def _gen_detector_pattern_from_fit(self, fit='best', generator='ideal'):
        # fit can be the best or last
        if fit == 'best':
            fit_obj = self.best_fit
        elif fit =='last':
            fit_obj = self.last_fit
        else:
            raise ValueError('parameter fit must be either \'best\' or \'last\'')

        # get values
        parameter_dict = fit_obj._parameters_dict.copy()
        dx = parameter_dict['dx']['value']
        dy = parameter_dict['dy']['value']
        phi = parameter_dict['phi']['value']
        total_events = parameter_dict['total_cts']['value'] if self._cost_function == 'chi2' else \
            np.sum(self.dp_pattern.matrixCurrent)
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
        sim_pattern = gen.make_pattern(dx, dy, phi, fractions_sims, total_events, sigma=sigma, type=generator)

        # Substitute only masked pixels that are in range (2.7° from center) and are not the chip edges
        # This can't really be made without keeping 2 set of masks, so all masked pixels are susbstituted.
        # This means some pixels with valid data but masked can still be susbtituted

        return sim_pattern



    def get_pattern_from_last_fit(self, normalization=None):
        fit_obj = self.last_fit
        assert isinstance(fit_obj, Fit)
        #print(fit_obj.sim_pattern.data)

        norm_factor = self._get_sim_normalization_factor(normalization, pattern_type=self._cost_function)

        dp = DataPattern(pattern_array=fit_obj.sim_pattern.data)
        dp._set_xymesh(fit_obj.XXmesh, fit_obj.YYmesh)
        dp.set_mask(fit_obj.sim_pattern.mask)
        return dp * norm_factor

    def get_pattern_from_best_fit(self, normalization=None):
        fit_obj = self.best_fit
        assert isinstance(fit_obj, Fit)
        #print(fit_obj.sim_pattern.data)

        norm_factor = self._get_sim_normalization_factor(normalization, pattern_type=self._cost_function)

        dp = DataPattern(pattern_array=fit_obj.sim_pattern.data)
        dp._set_xymesh(fit_obj.XXmesh, fit_obj.YYmesh)
        dp.set_mask(fit_obj.sim_pattern.mask)
        return dp * norm_factor

    def get_datapattern(self, normalization=None, substitute_masked_with=None, which_fit='last'):

        dp_pattern = copy.deepcopy(self.dp_pattern)

        if substitute_masked_with is not None:
            sim_pattern = self._gen_detector_pattern_from_fit(fit=which_fit, generator=substitute_masked_with)

            #if self._cost_function == 'ml':
            #    sim_pattern = sim_pattern * np.sum(dp_pattern.matrixCurrent)

            print('data\n', dp_pattern.matrixCurrent.data[dp_pattern.matrixCurrent.mask],
                'sim\n', sim_pattern.data[dp_pattern.matrixCurrent.mask])

            dp_pattern.matrixCurrent.data[dp_pattern.matrixCurrent.mask] = \
                sim_pattern.data[dp_pattern.matrixCurrent.mask]

            print('data\n', dp_pattern.matrixCurrent.data[dp_pattern.matrixCurrent.mask])

        norm_factor = self._get_sim_normalization_factor(normalization, pattern_type='data')

        return dp_pattern * norm_factor

