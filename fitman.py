#!/usr/bin/env python3

'''
Fit manager is the kernel class for fitting.
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

from read2dl.lib2dl import lib2dl
#from patterncreator import PatternCreator, create_detector_mesh
from MedipixMatrix import MedipixMatrix
from fits import fits

import pandas as pd
import os
import numpy as np
import math


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
            self.lib = library
        elif isinstance(library, str):
            if not os.path.isfile(library):
                raise ValueError('data is a str but filepath is not valid')
            else:
                self.lib = lib2dl(library)
        else:
            ValueError('data_pattern input error')

    def run_fits(self, *args, method='chi2', get_errors=False, fit_sigma=True):

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

        if len(patterns_list) == 1:
            patterns_list += ((None,), (None,),)
        elif len(patterns_list) == 2:
            patterns_list += ((None,),)

        print('args ', args)

        print('ar ', ar)

        print('patterns_list ', patterns_list)

        columns = ('value', 'x', 'x_err', 'y', 'y_err', 'phi', 'phi_err', 'counts', 'counts_err', 'sigma', 'sigma_err'
                   'fraction1', 'fraction1_err', 'fraction2', 'fraction2_err', 'fraction3', 'fraction3_err')
        self.df = pd.DataFrame(data=None, columns=columns)

        for p1 in patterns_list[0]:
            for p2 in patterns_list[1]:
                for p3 in patterns_list[2]:
                    print('P1, P2, P3 - ', p1, ', ', p2, ', ', p3)
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
                    append_dic['fraction1'] = ft.res['x'][3+di] if patterns_list[0][0] is not None else None
                    append_dic['fraction2'] = ft.res['x'][4+di] if patterns_list[1][0] is not None else None
                    append_dic['fraction3'] = ft.res['x'][5+di] if patterns_list[2][0] is not None else None
                    append_dic['value'] = ft.res['fun']
                    #print('append_dic ', append_dic)
                    self.df = self.df.append(append_dic, ignore_index=True)
                    #print('self.df ', self.df)

                    if ft.res['fun'] < self.min_value:
                        self.best_fit = ft

    def save_output(self, filename, save_figure=False):
        pass


if __name__ == '__main__':
    # filename
    filename = '/home/eric/Desktop/jsontest.json'

    # library
    library = '/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue646g26.2dl'

    # fits

    fm = fitman()
    fm.add_pattern(filename, library)
    P1 = np.array((0,))
    P2 = np.arange(0, 25) # 249
    fm.run_fits(P1, P2, method='chi2', get_errors=False, fit_sigma=True)

    # plot
    #plt.figure()
    #plt.plot(fm.df['value'])
    #plt.show()

    print(fm.df)

