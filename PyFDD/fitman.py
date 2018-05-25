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
    # pattern, library, pattern numbers, fit options
    def __init__(self):
        # Internal variable
        self.min_value = None
        self.fixed_values = {}
        self.keys = ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')

        # Output
        self.results = None

        # Stored objects
        self.best_fit = None
        self.mm_pattern = None
        self.lib = None

        # order of columns in results
        self.columns = \
            ['value', 'D.O.F.', 'x', 'x_err', 'y', 'y_err', 'phi', 'phi_err',
             'counts', 'counts_err', 'sigma', 'sigma_err',
             'site1 n', 'p1', 'site1 description', 'site1 factor', 'site1 u2', 'site1 fraction', 'fraction1_err',
             'site2 n', 'p2', 'site2 description', 'site2 factor', 'site2 u2', 'site2 fraction', 'fraction2_err',
             'site3 n', 'p3', 'site3 description', 'site3 factor', 'site3 u2', 'site3 fraction', 'fraction3_err']
        self.df = pd.DataFrame(data=None, columns=self.columns)

    def add_pattern(self,data_pattern, library):
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

    def set_fixed_values(self, **kwargs):
        #('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        self.fixed_values = {}
        for key in kwargs.keys():
            if key in self.keys:
                self.fixed_values[key] = kwargs[key]
            else:
                raise(ValueError, 'key word ' + key + 'is not recognized!')

    def _get_initial_values(self):
        #('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        p0 = ()
        p_fix = ()
        for key in self.keys:
            if key in self.fixed_values:
                p0 += (self.fixed_values[key],)
                p_fix += (True,)
            else:
                if key == 'dx':
                    p0 += (self.mm_pattern.center[0],)
                elif key == 'dy':
                    p0 += (self.mm_pattern.center[1],)
                elif key == 'phi':
                    p0 += (self.mm_pattern.angle,)
                elif key == 'total_cts':
                    patt = self.mm_pattern.matrixOriginal.copy()
                    #counts_ordofmag = 10 ** (int(math.log10(patt.sum())))
                    counts= patt.sum()
                    p0 += (counts,)
                elif key == 'sigma':
                    p0 += (0.1,)
                else:
                    # assuming a pattern fraction
                    p0 += (0.15,)
                p_fix += (False,)

        return p0, p_fix

    def _build_fits_obj(self, cost_func='chi2', optimization_profile='default', min_method='L-BFGS-B',
                        sub_pixels=1, p1=None, p2=None, p3=None, verbose_graphics=False):

        ft = fits(self.lib)
        ft.verbose_graphics = verbose_graphics

        ft.set_optimization_profile(optimization_profile, min_method)

        patt = self.mm_pattern.matrixCurrent.copy()
        xmesh = self.mm_pattern.xmesh.copy()
        ymesh = self.mm_pattern.ymesh.copy()

        ft.set_data_pattern(xmesh, ymesh, patt)

        # ignore similar patterns
        p1_fit = p2_fit = p3_fit = None
        p1_fit = p1
        if p2 is not None:
            p2_fit = p2 if not p2 == p1 else None
        if p3 is not None:
            p3_fit = p3 if not (p3 == p1 or p3 == p2) else None
        ft.set_patterns_to_fit(p1_fit, p2_fit, p3_fit)

        p0, p0_fix = self._get_initial_values()
        ft.parameters_dict['sub_pixels']['value'] = sub_pixels
        append_dic = {}
        if cost_func == 'chi2':
            counts_ordofmag = 10 ** (int(math.log10(patt.sum())))
            ft.set_scale_values(dx=1, dy=1, phi=1, total_cts=counts_ordofmag,
                                sigma=1, f_p1=1, f_p2=1, f_p3=1)
            ft.set_inicial_values(p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7])
            ft.fix_parameters(p0_fix[0], p0_fix[1], p0_fix[2], p0_fix[3], p0_fix[4], p0_fix[5],
                              p0_fix[6], p0_fix[7])

        if cost_func == 'ml':
            ft.set_scale_values(dx=1, dy=1, phi=1, total_cts=-1,
                                sigma=1, f_p1=1, f_p2=1, f_p3=1)
            ft.set_inicial_values(p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7])
            ft.fix_parameters(p0_fix[0], p0_fix[1], p0_fix[2], p0_fix[3], p0_fix[4], p0_fix[5],
                              p0_fix[6], p0_fix[7])

        return ft


    def _fill_results_dict(self, ft, cost_func, get_errors, p1=None, p2=None, p3=None):

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
        append_dic['counts'] = parameter_dict['total_cts']['value'] if cost_func == 'chi2' else np.nan
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
            append_dic['counts_err'] = parameter_dict['total_cts']['std'] if cost_func == 'chi2' else np.nan
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


    def run_fits(self, *args, cost_func='chi2', sub_pixels=1, optimization_profile='default',
                 min_method='L-BFGS-B', get_errors=False):

        assert isinstance(self.mm_pattern, MedipixMatrix)
        # each input is a range of patterns to fit
        assert isinstance(get_errors, bool)
        if cost_func not in ('chi2', 'ml'):
            raise ValueError('cost_func not valid. Use chi2 or ml')

        if get_errors is not False:
            raise warnings.warn('Errors and visualization are by default off in run_fits. Use run_single_fit')
            get_errors = False


        patterns_list = ()
        for ar in args:
            patterns_list += (np.array(ar),)
        assert len(patterns_list) >= 1

        if len(patterns_list) == 1:
            patterns_list += ((None,), (None,),)
        elif len(patterns_list) == 2:
            patterns_list += ((None,),)

        #print('args ', args)

        #print('ar ', ar)

        #print('patterns_list ', patterns_list)

        for p1 in patterns_list[0]:
            for p2 in patterns_list[1]:
                for p3 in patterns_list[2]:
                    print('P1, P2, P3 - ', p1, ', ', p2, ', ', p3)

                    # errors and visualization are by default off in run_fits
                    self.run_single_fit(p1, p2, p3, cost_func, sub_pixels, optimization_profile, min_method=min_method,
                                        verbose_graphics=False, get_errors=get_errors)


    def run_single_fit(self, p1, p2=None, p3=None, cost_func='chi2', sub_pixels=1,
                       optimization_profile='default', min_method='L-BFGS-B',
                       verbose_graphics=False, get_errors=False):

        assert isinstance(self.mm_pattern, MedipixMatrix)
        # each input is a range of patterns to fit
        assert isinstance(verbose_graphics, bool)
        assert isinstance(get_errors, bool)
        if cost_func not in ('chi2', 'ml'):
            raise ValueError('cost_func not valid. Use chi2 or ml')

        ft = self._build_fits_obj(cost_func, optimization_profile, min_method, sub_pixels,
                                  p1, p2, p3)

        ft.verbose_graphics = verbose_graphics

        ft.minimize_cost_function(cost_func)
        print(ft.results)

        if get_errors:
            ft.get_std_from_hessian(ft.results['x'], func='cost_func')

        self._fill_results_dict(ft, cost_func, get_errors, p1, p2, p3)

        if self.min_value is None:
            self.best_fit = ft
            self.min_value = ft.results['fun']
        elif ft.results['fun'] < self.min_value:
           self.best_fit = ft
           self.min_value = ft.results['fun']


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




if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    # filename
    filename = '/home/eric/Desktop/jsontest.json'

    # library
    library = '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue646g26.2dl'

    # fits

    fm = fitman()
    fm.add_pattern(filename, library)

    P1 = np.array((1,))
    P2 = np.arange(1, 3) # 249
    fm.set_fixed_values(sigma=0)
    fm.run_fits(P1, P2, cost_func='chi2', get_errors=True, sub_pixels=1)

    #fm.save_output('/home/eric/Desktop/test_fit.xls', save_figure=True)

    # plot
    #plt.figure()
    #plt.plot(fm.df['value'])
    #plt.show()

    print(fm.df)

