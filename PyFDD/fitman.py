#!/usr/bin/env python3

'''
Fit manager is the kernel class for fitting.
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

from .lib2dl import lib2dl
#from patterncreator import PatternCreator, create_detector_mesh
from .MedipixMatrix import MedipixMatrix
from .fits import fits

import pandas as pd
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings


class fitman:
    '''
    The class fitman is a helper class for using fits in pyfdd.
    You should be able to do all standard routine analysis from fitman.
    It also help in creating graphs, using fit options and saving results.
    '''
    def __init__(self, cost_function='chi2', sub_pixels=1):
        '''
        fitman is a helper class for using fits in pyfdd.
        :param cost_function: The type of cost function to use. Possible values are 'chi2' for chi-square
        and 'ml' for maximum likelihood.
        :param sub_pixels: The number of subpixels to integrate during fit in x and y.
        '''

        if cost_function not in ('chi2', 'ml'):
            raise ValueError('cost_function not valid. Use chi2 or ml')

        if not isinstance(sub_pixels, int):
            raise ValueError('sub_pixels must be of type int')

        self.done_param_verbose = False

        # Output
        self.results = None

        # Stored objects
        self.min_value = None
        self.best_fit = None
        self.last_fit = None
        self.mm_pattern = None
        self.lib = None

        # Fit settings
        self.keys = ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        self._cost_function = cost_function
        self._fit_options = {}
        self._fit_options_profile = 'default'
        self._minimization_method = 'L-BFGS-B'
        self.set_minimization_settings()
        self._sub_pixels = sub_pixels
        # total_cts is overwriten with values from the data pattern
        self._scale = {'dx':1, 'dy':1, 'phi':1, 'total_cts':1,
                       'sigma':1, 'f_p1':1, 'f_p2':1, 'f_p3':1}
        self._bounds = {'dx': (-3, +3), 'dy': (-3, +3), 'phi': (None, None), 'total_cts': (1, None),
                         'sigma': (0.01, None), 'f_p1': (0, 1), 'f_p2': (0, 1), 'f_p3': (0, 1)}

        # Fit parameters settings
        # overwrite defaults from fits
        self.p_initial_values = {}
        self.p_fixed_values = {}

        # order of columns in results
        self.columns = \
            ['value', 'D.O.F.', 'x', 'x_err', 'y', 'y_err', 'phi', 'phi_err',
             'counts', 'counts_err', 'sigma', 'sigma_err',
             'site1 n', 'p1', 'site1 description', 'site1 factor', 'site1 u2', 'site1 fraction', 'fraction1_err',
             'site2 n', 'p2', 'site2 description', 'site2 factor', 'site2 u2', 'site2 fraction', 'fraction2_err',
             'site3 n', 'p3', 'site3 description', 'site3 factor', 'site3 u2', 'site3 fraction', 'fraction3_err']
        self.df = pd.DataFrame(data=None, columns=self.columns)

    def set_pattern(self, data_pattern, library):
        '''
        Set the pattern to fit.
        :param data_pattern: path or MedipixMatrix
        :param library: path or lib2dl
        '''
        if isinstance(data_pattern, MedipixMatrix):
            # all good
            self.mm_pattern = data_pattern
        elif isinstance(data_pattern,  str):
            if not os.path.isfile(data_pattern):
                raise ValueError('data is a str but filepath is not valid')
            else:
                self.mm_pattern = MedipixMatrix(file_path=data_pattern)
        else:
            ValueError('data_pattern input error')

        if isinstance(library, lib2dl):
            # all good
            self.lib = library
        elif isinstance(library, str):
            if not os.path.isfile(library):
                raise ValueError('data is a str but filepath is not valid')
            else:
                self.lib = lib2dl(library)
        else:
            ValueError('data_pattern input error')

        print('\nMedipix pattern added')
        print('Inicial orientation (x, y, phi) is (',
              self.mm_pattern.center[0], ', ', self.mm_pattern.center[1], ',',
              self.mm_pattern.angle, ')')

    def _print_settings(self, ft):
        '''
        prints the settings that are in use during fit
        :param ft: fits object
        '''
        assert isinstance(ft, fits)
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
        for key in self.keys:
            # {'p0':None, 'value':None, 'use':False, 'std':None, 'scale':1, 'bounds':(None,None)}
            print(string_temp.format(
                key,
                ft.parameters_dict[key]['p0'],
                ft.parameters_dict[key]['use'] == False,
                '({},{})'.format(ft.parameters_dict[key]['bounds'][0],ft.parameters_dict[key]['bounds'][1]),
                ft.parameters_dict[key]['scale']
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
            if key in self.keys:
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
            if key in self.keys:
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
            if key in self.keys:
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
            if key in self.keys:
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
        for key in self.keys:
            # Use user defined fixed value
            if key in self.p_fixed_values:
                p0 += (self.p_fixed_values[key],)
                p_fix += (True,)
            # Use user defined initial value
            elif key in self.p_initial_values:
                p0 += (self.p_initial_values[key],)
                p_fix += (False,)
            # Use fitman choice
            else:
                if key == 'dx':
                    p0 += (p0_last[p0_last_i],) if p0_pass else (self.mm_pattern.center[0],)
                elif key == 'dy':
                    p0 += (p0_last[p0_last_i],) if p0_pass else (self.mm_pattern.center[1],)
                elif key == 'phi':
                    p0 += (p0_last[p0_last_i],) if p0_pass else (self.mm_pattern.angle,)
                elif key == 'total_cts':
                    patt = self.mm_pattern.matrixOriginal.copy()
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
        for key in self.keys:
            # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
            # total_cts is a spacial case at it uses the counts from the pattern
            if key == 'total_cts':
                if self._cost_function == 'chi2':
                    patt = self.mm_pattern.matrixCurrent
                    counts_ordofmag = 10 ** (int(math.log10(patt.sum())))
                    scale += (counts_ordofmag * self._scale[key],)
                elif self._cost_function == 'ml':
                    scale += (-1,)
            else:
                scale += (self._scale[key],)
        return scale

    def _build_fits_obj(self, p1=None, p2=None, p3=None, verbose_graphics=False, pass_results=False):
        '''
        Builds a fits object
        :param p1: pattern 1
        :param p2: pattern 2
        :param p3: pattern 3
        :param verbose_graphics: plot pattern as it is being fit
        :param pass_results: argument for _get_initial_values
        :return: fits object
        '''

        ft = fits(self.lib)
        ft.verbose_graphics = verbose_graphics

        ft.set_sub_pixels(self._sub_pixels)
        ft.set_fit_options(self._fit_options)

        patt = self.mm_pattern.matrixCurrent
        xmesh = self.mm_pattern.xmesh
        ymesh = self.mm_pattern.ymesh
        ft.set_data_pattern(xmesh, ymesh, patt)

        # ignore similar patterns
        p1_fit = p2_fit = p3_fit = None
        p1_fit = p1
        if p2 is not None:
            p2_fit = p2 if not p2 == p1 else None
        if p3 is not None:
            p3_fit = p3 if not (p3 == p1 or p3 == p2) else None
        ft.set_patterns_to_fit(p1_fit, p2_fit, p3_fit)

        # Get initial values
        p0, p0_fix = self._get_initial_values(pass_results=pass_results)

        # Get scale
        scale = self._get_scale_values()

        ft.set_scale_values(dx=scale[0], dy=scale[1], phi=scale[2], total_cts=scale[3],
                            sigma=scale[4], f_p1=scale[5], f_p2=scale[6], f_p3=scale[7])

        ft.set_inicial_values(p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7])

        ft.fix_parameters(p0_fix[0], p0_fix[1], p0_fix[2], p0_fix[3], p0_fix[4], p0_fix[5],
                          p0_fix[6], p0_fix[7])

        ft.set_bound_values(dx=self._bounds['dx'], dy=self._bounds['dy'], phi=self._bounds['phi'],
                            total_cts=self._bounds['total_cts'], sigma=self._bounds['sigma'],
                            f_p1=self._bounds['f_p1'], f_p2=self._bounds['f_p2'], f_p3=self._bounds['f_p3'])

        return ft


    def _fill_results_dict(self, ft, get_errors, p1=None, p2=None, p3=None):

        assert isinstance(ft, fits), "ft is not of type PyFDD.fits."

        patt = self.mm_pattern.matrixCurrent.copy()

        # keys are 'pattern_1','pattern_2','pattern_3','sub_pixels','dx','dy','phi',
        # 'total_cts','sigma','f_p1','f_p2','f_p3'
        parameter_dict = ft.parameters_dict.copy()
        append_dic = {}
        append_dic['value'] = ft.results['fun']
        append_dic['D.O.F.'] = np.sum(~patt.mask)
        append_dic['x'] = parameter_dict['dx']['value']
        append_dic['y'] = parameter_dict['dy']['value']
        append_dic['phi'] = parameter_dict['phi']['value']
        append_dic['counts'] = parameter_dict['total_cts']['value'] if self._cost_function == 'chi2' else np.nan
        append_dic['sigma'] = parameter_dict['sigma']['value']
        if p1 is not None:
            append_dic['site1 n'] = self.lib.ECdict["Spectrums"][p1 - 1]["Spectrum number"]
            append_dic['p1'] = p1
            append_dic['site1 description'] = self.lib.ECdict["Spectrums"][p1 - 1]["Spectrum_description"]
            append_dic['site1 factor'] = self.lib.ECdict["Spectrums"][p1 - 1]["factor"]
            append_dic['site1 u2'] = self.lib.ECdict["Spectrums"][p1 - 1]["u2"]
            append_dic['site1 fraction'] = parameter_dict['f_p1']['value']
        if p2 is not None:
            append_dic['site2 n'] = self.lib.ECdict["Spectrums"][p2 - 1]["Spectrum number"]
            append_dic['p2'] = p2
            append_dic['site2 description'] = self.lib.ECdict["Spectrums"][p2 - 1]["Spectrum_description"]
            append_dic['site2 factor'] = self.lib.ECdict["Spectrums"][p2 - 1]["factor"]
            append_dic['site2 u2'] = self.lib.ECdict["Spectrums"][p2 - 1]["u2"]
            if not p2 == p1:
                append_dic['site2 fraction'] = parameter_dict['f_p2']['value']
        if p3 is not None:
            append_dic['site3 n'] = self.lib.ECdict["Spectrums"][p3 - 1]["Spectrum number"]
            append_dic['p3'] = p3
            append_dic['site3 description'] = self.lib.ECdict["Spectrums"][p3 - 1]["Spectrum_description"]
            append_dic['site3 factor'] = self.lib.ECdict["Spectrums"][p3 - 1]["factor"]
            append_dic['site3 u2'] = self.lib.ECdict["Spectrums"][p3 - 1]["u2"]
            if not (p3 == p1 or p3 == p2):
                append_dic['site3 fraction'] = parameter_dict['f_p3']['value']
        if get_errors:
            append_dic['x_err'] = parameter_dict['dx']['std']
            append_dic['y_err'] = parameter_dict['dy']['std']
            append_dic['phi_err'] = parameter_dict['phi']['std']
            append_dic['counts_err'] = parameter_dict['total_cts']['std'] if self._cost_function == 'chi2' else np.nan
            append_dic['sigma_err'] = parameter_dict['sigma']['std']
            append_dic['fraction1_err'] = parameter_dict['f_p1']['std'] if p1 is not None else np.nan
            append_dic['fraction2_err'] = parameter_dict['f_p2']['std'] if p2 is not None and \
                                                                                not p2 == p1 \
                                                                                else np.nan
            append_dic['fraction3_err'] = parameter_dict['f_p3']['std'] if p3 is not None and \
                                                                                not (p3 == p1 or p3 == p2) \
                                                                                else np.nan

        # print('append_dic ', append_dic)
        self.df = self.df.append(append_dic, ignore_index=True)
        self.df = self.df[self.columns]
        # print('self.df ', self.df)


# TODO
#cost_func='chi2', sub_pixels=1,
#                    optimization_profile='default', min_method='L-BFGS-B',

    def run_fits(self, *args, pass_results=False, verbose=1):
        '''
        Run fits for a list of sites.
        :param args: list of patterns for each site. Up to tree sites are possible
        :param pass_results: Use the last fit parameter results as input for the next.
        :param verbose: 0 silent, 1 default and 2 max verbose
        :return:
        '''

        assert isinstance(self.mm_pattern, MedipixMatrix)

        self.done_param_verbose = False

        patterns_list = ()
        for ar in args:
            patterns_list += (np.array(ar),)
        assert len(patterns_list) >= 1

        if len(patterns_list) == 1:
            patterns_list += ((None,), (None,),)
        elif len(patterns_list) == 2:
            patterns_list += ((None,),)

        for p1 in patterns_list[0]:
            for p2 in patterns_list[1]:
                for p3 in patterns_list[2]:

                    # errors and visualization are by default off in run_fits
                    self._single_fit(p1, p2, p3, verbose=verbose, pass_results=pass_results)

    def run_single_fit(self, p1, p2=None, p3=None, verbose=1,
                       verbose_graphics=False, get_errors=False):

        self.done_param_verbose = False

        self._single_fit(p1, p2, p3, get_errors=get_errors, pass_results=False,
                         verbose=verbose, verbose_graphics=verbose_graphics)


    def _single_fit(self, p1, p2=None, p3=None, get_errors=False, pass_results=False,
                    verbose=1, verbose_graphics=False):

        assert isinstance(self.mm_pattern, MedipixMatrix)
        # each input is a range of patterns to fit
        assert isinstance(verbose_graphics, bool)
        assert isinstance(get_errors, bool)

        ft = self._build_fits_obj(p1, p2, p3, verbose_graphics, pass_results=pass_results)

        if verbose > 0 and self.done_param_verbose is False:
            self._print_settings(ft)

        if verbose > 0:
            print('P1, P2, P3 - ', p1, ', ', p2, ', ', p3)

        ft.minimize_cost_function(self._cost_function)

        if verbose > 1:
            print(ft.results)

        if get_errors:
            ft.get_std_from_hessian(ft.results['x'], func='cost_func')

        self._fill_results_dict(ft, get_errors, p1, p2, p3)

        # Keep best fit
        if self.min_value is None:
            self.best_fit = ft
            self.min_value = ft.results['fun']
        elif ft.results['fun'] < self.min_value:
           self.best_fit = ft
           self.min_value = ft.results['fun']

        self.last_fit = ft

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
