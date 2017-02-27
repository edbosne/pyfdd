#!/usr/bin/env python3

'''
Fit manager is the kernel class for fitting.
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

from lib2dl import lib2dl
from patterncreator import *
from MedipixMatrix import *
from fits import fits
import pandas as pd

import os

class fitman:
    # pattern, library, pattern numbers, fit options
    def __init__(self):
        # Internal variable
        self.min_value = 10**12

        # Output
        self.results = None

        # Stored objects
        self.best_fit = None
        self.mm_pattern = None
        self.lib = None

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
            self.lib = lib2dl
        elif isinstance(library, str):
            if not os.path.isfile(library):
                raise ValueError('data is a str but filepath is not valid')
            else:
                self.mm_pattern = MedipixMatrix(file_path=data_pattern)
        else:
            ValueError('data_pattern input error')

    def run_fits(self, method='chi2', get_errors=False, fit_sigma=True, *args):

        # each input is a range of patterns to fit
        assert isinstance(get_errors, bool)
        if method not in ('chi2', 'ml'):
            raise ValueError('method not valid. Use chi2 or ml')

        patt = self.mm_pattern.matrixOriginal.copy()
        xmesh = self.mm_pattern.xmesh.copy()
        ymesh = self.mm_pattern.ymesh.copy()
        counts_ordofmag = 10 ** (int(math.log10(patt.sum())))

        patterns_list = ()
        for ar in args:
            patterns_list += (np.array(ar),)
        assert len(ar) >= 1

        if len(ar) == 1:
            ar += (None, None,)
        elif len(ar) == 2:
            ar += (None,)

        columns = ('x', 'x_err', 'y', 'y_err', 'phi', 'phi_err', 'counts', 'counts_err', 'sigma', 'sigma_err'
                   'fraction1', 'fraction1_err', 'fraction2', 'fraction2_err', 'fraction3', 'fraction3_err')
        df = pd.DataFrame(data=None, columns=columns)

        for p1 in patterns_list[0]:
            for p2 in patterns_list[1]:
                for p3 in patterns_list[2]:
                    ft = fits(self.lib)
                    ft.set_data_pattern(xmesh, ymesh, patt)
                    ft.set_patterns_to_fit(p1, p2, p3)
                    ft.fit_sigma = fit_sigma
                    append_dic = {}
                    if method == 'chi2':
                        ft.set_scale_values(dx=1, dy=1, phi=1, total_cts=counts_ordofmag,
                                            sigma=1, f_p1=1, f_p2=1, f_p3=1)
                        ft.set_inicial_values(self.mm_pattern.center[0],
                                              self.mm_pattern.center[1],
                                              self.mm_pattern.angle,
                                              counts_ordofmag, sigma=0.1)
                        ft.minimize_chi2()
                        # TODO get errors
                    if method == 'ml':
                        ft.set_scale_values(dx=1, dy=1, phi=1, total_cts=-1,
                                            sigma=1, f_p1=1, f_p2=1, f_p3=1)
                        ft.set_inicial_values(self.mm_pattern.center[0],
                                              self.mm_pattern.center[1],
                                              self.mm_pattern.angle,
                                              counts_ordofmag, sigma=0.1)
                        ft.maximize_likelyhood()
                        # TODO get errors
                    append_dic['x'] = ft.res['x'][0]
                    append_dic['y'] = ft.res['x'][1]
                    append_dic['phi'] = ft.res['x'][2]
                    append_dic['counts'] = ft.res['x'][3] if method == 'chi2' else None
                    di = 1 if  method == 'chi2' else 0
                    append_dic['sigma'] = ft.res['x'][3+di] if fit_sigma else None
                    di += 1 if fit_sigma else 0
                    append_dic['fraction1'] = ft.res['x'][3+di] if patterns_list[0] is not None else None
                    append_dic['fraction2'] = ft.res['x'][4+di] if patterns_list[1] is not None else None
                    append_dic['fraction3'] = ft.res['x'][5+di] if patterns_list[2] is not None else None
                    df.append(append_dic, ignore_index=True)

                    if ft.res['fun'] < self.min_value:
                        self.best_fit = ft.copy()

    def save_output(self,filename):
        pass

    def get_pd_table(self):
        pass
