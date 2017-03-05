#!/usr/bin/env python3

'''
The fits object gets access to a lib2dl object and performs fits and statistical tests to data or MC simulation
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

from read2dl.lib2dl import lib2dl
from patterncreator import *
from MedipixMatrix import *

import numpy as np
import scipy.optimize as op
import scipy.stats as st
import matplotlib.pyplot as plt
import math
#import numdifftools as nd
from scipy.ndimage import gaussian_filter



class fits:
    def __init__(self,lib):
        assert isinstance(lib,lib2dl)
        self.lib = lib
        self.pattern_1_n = 0
        self.pattern_1_use = True
        self.pattern_2_n = None
        self.pattern_2_use = False
        self.pattern_3_n = None
        self.pattern_3_use = False
        self.n_events = None
        self.n_events_set = False
        self.fit_n_events = False
        self.XXmesh = None
        self.YYmesh = None
        self.data_pattern = None
        self.sim_pattern = None
        self.data_pattern_is_set = False
        self.p0 = (None,)
        self.p0_scale = np.ones((8))
        self.fit_sigma = False
        self.res = None

    def set_data_pattern(self,XXmesh,YYmesh,pattern,mask=None):
        self.XXmesh = XXmesh.copy()
        self.YYmesh = YYmesh.copy()
        self.data_pattern = pattern.copy()
        self.data_pattern_is_set = True
        self.n_events = self.data_pattern.sum()
        self.n_events_set = True

    def set_patterns_to_fit(self,p1_n=None,p2_n=None,p3_n=None):
        self.pattern_1_n = p1_n
        self.pattern_1_use = False if p1_n == None else True
        self.pattern_2_n = p2_n
        self.pattern_2_use = False if p2_n == None else True
        self.pattern_3_n = p3_n
        self.pattern_3_use = False if p3_n == None else True

    def set_inicial_values(self,dx=1,dy=1,phi=5,total_cts=1, sigma=0, f_p1=0.25,f_p2=0.25,f_p3=0.25):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        p0 = (dx/self.p0_scale[0],)
        p0 += (dy/self.p0_scale[1],)
        p0 += (phi/self.p0_scale[2],)
        p0 += (total_cts / self.p0_scale[3],) if total_cts > 0 else ()
        p0 += (sigma / self.p0_scale[4],) if self.fit_sigma > 0 else ()
        di = 1 if self.fit_sigma else 0
        p0 += (f_p1/self.p0_scale[4+di],) if self.pattern_1_use else ()
        p0 += (f_p2/self.p0_scale[5+di],) if self.pattern_2_use else ()
        p0 += (f_p3/self.p0_scale[6+di],) if self.pattern_3_use else ()
        self.p0 = np.array(p0)
        print('p0 - ', p0)

    def set_scale_values(self, dx=1, dy=1, phi=1, total_cts=1, sigma=1, f_p1=1, f_p2=1, f_p3=1):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        p0_scale = (dx,)
        p0_scale += (dy,)
        p0_scale += (phi,)
        p0_scale += (total_cts,) if total_cts > 0 else ()
        p0_scale += (sigma,) if self.fit_sigma > 0 else ()
        p0_scale += (f_p1,)
        p0_scale += (f_p2,)
        p0_scale += (f_p3,)
        self.p0_scale = np.array(p0_scale)

    def print_variance(self,x,var):
        # TODO add number of events
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        params = x
        dx = params[0]
        d_dx = var[0]
        dy = params[1]
        d_dy = var[1]
        phi = params[2]
        d_phi = var[2]
        N_rand = params[3]
        d_N_rand = var[3]
        N_p1 = params[4] if self.pattern_1_use else 0
        d_N_p1 = var[4] if self.pattern_1_use else 0
        N_p2 = params[5] if self.pattern_2_use else 0
        d_N_p2 = var[5] if self.pattern_2_use else 0
        N_p3 = params[6] if self.pattern_3_use else 0
        d_N_p3 = var[6] if self.pattern_3_use else 0

        total_f = N_rand + N_p1 + N_p2 + N_p3
        f_rand = N_rand / total_f
        f_1 = N_p1 / total_f
        f_2 = N_p2 / total_f
        f_3 = N_p3 / total_f
        print('rand', N_rand, d_N_rand)
        print('p1', N_p1, d_N_p1)
        d_f_rand = np.abs(d_N_rand / total_f \
                          - N_rand * (
                          d_N_rand / total_f ** 2 + d_N_p1 / total_f ** 2 + d_N_p2 / total_f ** 2 + d_N_p3 / total_f ** 2))
        d_f_1 = np.abs(d_N_p1 / total_f \
                       - N_p1 * (
                       d_N_rand / total_f ** 2 + d_N_p1 / total_f ** 2 + d_N_p2 / total_f ** 2 + d_N_p3 / total_f ** 2))
        d_f_2 = np.abs(d_N_p2 / total_f \
                       - N_p2 * (
                       d_N_rand / total_f ** 2 + d_N_p1 / total_f ** 2 + d_N_p2 / total_f ** 2 + d_N_p3 / total_f ** 2))
        d_f_3 = np.abs(d_N_p3 / total_f \
                       - N_p3 * (
                       d_N_rand / total_f ** 2 + d_N_p1 / total_f ** 2 + d_N_p2 / total_f ** 2 + d_N_p3 / total_f ** 2))

        res = {'dx': dx, 'd_dx': d_dx,
               'dy': dy, 'd_dy': d_dy,
               'phi': phi, 'd_phi': d_phi,
               'f_rand': f_rand, 'd_f_rand': d_f_rand,
               'f_1': f_1, 'd_f_1': d_f_1,
               'f_2': f_2, 'd_f_2': d_f_2,
               'f_3': f_3, 'd_f_3': d_f_3}

        print(('dx     = {dx:.4f} +- {d_dx:.4f}\n' +
               'dy     = {dy:.4f} +- {d_dy:.4f}\n' +
               'phi    = {phi:.4f} +- {d_phi:.4f}\n' +
               'f_rand = {f_rand:.4f} +- {d_f_rand:.4f}\n' +
               'f_1    = {f_1:.4f} +- {d_f_1:.4f}\n' +
               'f_2    = {f_2:.4f} +- {d_f_2:.4f}\n' +
               'f_3    = {f_3:.4f} +- {d_f_3:.4f}').format(**res))

        return res

# methods for scipy.optimize.curve_fit
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit
    def simulation_callable(self, x, *params):
        x = np.array(x)
        # separate parameters
        print(params)
        dx = params[0]
        dy = params[1]
        phi = params[2]
        N_rand = params[3]
        N_p1 = params[4] if self.pattern_1_use else 0
        N_p2 = params[5] if self.pattern_2_use else 0
        N_p3 = params[6] if self.pattern_3_use else 0
        # get patterns
        p1 = self.pattern_1_n if self.pattern_1_use else None
        p2 = self.pattern_2_n if self.pattern_2_use else None
        p3 = self.pattern_3_n if self.pattern_3_use else None
        # convert x to XXmesh and YYmesh
        print('x shape',x.shape)
        XXmesh = np.reshape(x[:,0],self.XXmesh.shape)
        YYmesh = np.reshape(x[:,1],self.YYmesh.shape)
        # call lib.get_patterns() lib.move_rotate(0.5,-0.5,0) lib.grid_interpolation(XXmesh,YYmesh)
        self.lib.get_patterns(N_rand, p1, N_p1, p2, N_p2, p3, N_p3)
        self.lib.move_rotate(dx, dy, phi)
        # this scales the fractions with total events
        sim_pattern = self.lib.grid_interpolation(XXmesh,YYmesh)
        sim_pattern *= self.n_events/sim_pattern.size

        # plt.figure(2)
        # plt.contourf(XXmesh)
        # plt.colorbar()
        # plt.figure(1)
        # plt.contourf(XXmesh,YYmesh,sim_pattern)
        # plt.colorbar()
        # plt.show()

        return sim_pattern.reshape(-1)

    def call_curve_fit(self):

        xdata = np.concatenate([self.XXmesh.reshape(-1)[np.newaxis].T,\
                                self.YYmesh.reshape(-1)[np.newaxis].T],1)
        ydata = self.data_pattern.reshape(-1)
        p0 = self.p0
        sigma = np.sqrt(ydata)

        return op.curve_fit(self.simulation_callable,xdata,ydata,p0=p0,sigma=sigma,method='lm')

# methods for chi-square minimization
    def chi_square_fun(self, experimental_data, simlulation_data):
        # delta degrees of freedom
        # dx, dy, phi
        ddof = 3
        ddof += 1 if self.pattern_1_use else 0
        ddof += 1 if self.pattern_2_use else 0
        ddof += 1 if self.pattern_2_use else 0
        return st.chisquare(experimental_data, simlulation_data,ddof,axis=None)

    def chi_square(self, dx, dy, phi, total_events, simulations, fractions_sims, sigma=0):
        """
        Calculates the Pearson chi2 for the given conditions.
        :param dx: delta x in angles
        :param dy: delta y in angles
        :param phi: delta phi in anlges
        :param total_events: total number of events
        :param simulations: simulations id number
        :param fractions_sims: fractions of each simulated pattern
        :return: Pearson's chi2
        """
        # set data pattern
        data_pattern = self.data_pattern.copy()
        if not len(simulations) == len(fractions_sims):
            raise ValueError("size o simulations is diferent than size o events")
        fractions_sims = np.array(fractions_sims)
        rnd_events = np.array([1 - fractions_sims.sum()])
        # generate sim pattern
        gen = PatternCreator(self.lib, self.XXmesh, self.YYmesh, simulations, mask=data_pattern.mask)
        fractions = np.concatenate((rnd_events, fractions_sims))
        sim_pattern = gen.make_pattern(dx, dy, phi, fractions, total_events, sigma=sigma, type='ideal')
        self.sim_pattern = sim_pattern.copy()
        # chi2, pval = self.chi_square_fun(data_pattern,sim_pattern)
        chi2 = np.sum((data_pattern - sim_pattern) ** 2 / np.abs(sim_pattern))
        #print('chi2 - ', chi2)
        # print('p-value - ',pval)
        # =====
        # fg = plt.figure(1)
        # ax = fg.add_subplot(111)
        # plt.ion()
        # cont = None
        # plt.contourf(self.XXmesh, self.YYmesh, (data_pattern-sim_pattern))
        # fg.canvas.draw()
        # plt.show(block=False)
        # =====
        return chi2

    def chi_square_call(self, params, enable_scale=False):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        # print('params ', params)
        p0_scale = self.p0_scale.copy() if enable_scale else np.ones(len(params))
        #print('p0_scale ', p0_scale)
        dx = params[0] * p0_scale[0]
        dy = params[1] * p0_scale[1]
        phi = params[2] * p0_scale[2]
        total_cts = (params[3] * p0_scale[3],)  # total counts
        sigma = (params[4] * p0_scale[4],)[0] if self.fit_sigma else 0
        di = 1 if self.fit_sigma else 0
        fractions_sims = ()
        fractions_sims += (params[4+di] * p0_scale[4+di],) if self.pattern_1_use else () # pattern 1
        fractions_sims += (params[5+di] * p0_scale[5+di],) if self.pattern_2_use else () # pattern 2
        fractions_sims += (params[6+di] * p0_scale[6+di],) if self.pattern_3_use else () # pattern 3
        # get patterns
        simulations  = (self.pattern_1_n,) if self.pattern_1_use else ()
        simulations += (self.pattern_2_n,) if self.pattern_2_use else ()
        simulations += (self.pattern_3_n,) if self.pattern_3_use else ()
        return self.chi_square(dx, dy, phi, total_cts, simulations, fractions_sims=fractions_sims, sigma=sigma)

    def minimize_chi2(self):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        p0 = self.p0
        #print('p0 - ', p0)
        bnds = ((-3,+3), (-3,+3), (None,None), (0, None))
        bnds += ((None, None),) if self.fit_sigma else ()
        bnds += ((0, 1),) if self.pattern_1_use else ()
        bnds += ((0, 1),) if self.pattern_2_use else ()
        bnds += ((0, 1),) if self.pattern_3_use else ()

        res = op.minimize(self.chi_square_call, p0, args=True, method='L-BFGS-B', bounds=bnds,\
                           options={'disp':False, 'maxiter':30, 'ftol':1e-7,'maxcor':1000}) #'eps':0.001,
        if self.fit_sigma:
            if self.pattern_1_use:
                if self.pattern_2_use:
                    if self.pattern_3_use:
                        res['x'] *= self.p0_scale[0:8]
                    else:
                        res['x'] *= self.p0_scale[0:7]
                else:
                    res['x'] *= self.p0_scale[0:6]
            else:
                res['x'] *= self.p0_scale[0:5]
        else:
            if self.pattern_1_use:
                if self.pattern_2_use:
                    if self.pattern_3_use:
                        res['x'] *= self.p0_scale[0:7]
                    else:
                        res['x'] *= self.p0_scale[0:6]
                else:
                    res['x'] *= self.p0_scale[0:5]
            else:
                res['x'] *= self.p0_scale[0:4]
        self.res = res

# methods for maximum likelihood
    def log_likelihood(self, dx, dy, phi, simulations, fractions_sims, sigma=0):
        """
        Calculates the Pearson chi2 for the given conditions.
        :param dx: delta x in angles
        :param dy: delta y in angles
        :param phi: delta phi in anlges
        :param simulations: simulations id number
        :param fractions_sims: fractions of each simulated pattern
        :return: likelihood
        """
        # set data pattern
        data_pattern = self.data_pattern.copy()
        if not len(simulations) == len(fractions_sims):
            raise ValueError("size o simulations is diferent than size o events")
        total_events = 1
        fractions_sims = np.array(fractions_sims)
        rnd_events = np.array([1 - fractions_sims.sum()])
        # generate sim pattern
        gen = PatternCreator(self.lib, self.XXmesh, self.YYmesh, simulations, mask=data_pattern.mask)
        fractions = np.concatenate((rnd_events, fractions_sims))
        sim_pattern = gen.make_pattern(dx, dy, phi, fractions, total_events, sigma=sigma, type='ideal')
        self.sim_pattern = sim_pattern.copy()
        # log likelihood
        ll = np.sum(data_pattern * np.log(sim_pattern))
        # extended log likelihood - no need to fit events
        #ll = -np.sum(events_per_sim) + np.sum(data_pattern * np.log(sim_pattern))
        #print('likelihood - ', ll)
        # =====
        # fg = plt.figure(1)
        # ax = fg.add_subplot(111)
        # plt.ion()
        # cont = None
        # plt.contourf(self.XXmesh, self.YYmesh, sim_pattern)
        # fg.canvas.draw()
        # plt.show(block=False)
        # =====
        return -ll

    def log_likelihood_call(self, params, enable_scale=False):
        #print(params)
        #print('params ', params)
        p0_scale = self.p0_scale.copy() if enable_scale else np.ones(len(params))
        #print(p0_scale)
        dx = params[0] * p0_scale[0]
        dy = params[1] * p0_scale[1]
        phi = params[2] * p0_scale[2]
        sigma = (params[3] * p0_scale[3],)[0] if self.fit_sigma else 0
        di = 1 if self.fit_sigma else 0
        fractions_sims = ()
        fractions_sims += (params[3+di] * p0_scale[3+di],) if self.pattern_1_use else () # pattern 1
        fractions_sims += (params[4+di] * p0_scale[4+di],) if self.pattern_2_use else () # pattern 2
        fractions_sims += (params[5+di] * p0_scale[5+di],) if self.pattern_3_use else () # pattern 3
        # get patterns
        simulations  = (self.pattern_1_n,) if self.pattern_1_use else ()
        simulations += (self.pattern_2_n,) if self.pattern_2_use else ()
        simulations += (self.pattern_3_n,) if self.pattern_3_use else ()
        return self.log_likelihood(dx, dy, phi, simulations, fractions_sims, sigma=sigma)

    def maximize_likelyhood(self):
        #print('p0 - ', self.p0)
        bnds = ((-3, +3), (-3, +3), (None, None)) #(self.p0[2]-5/self.p0_scale[2],self.p0[2]+5/self.p0_scale[2])) # no need for number of cts
        bnds += ((None, None),) if self.fit_sigma else ()
        bnds += ((0, 1),) if self.pattern_1_use else ()
        bnds += ((0, 1),) if self.pattern_2_use else ()
        bnds += ((0, 1),) if self.pattern_3_use else ()
        #print('self.p0', self.p0)
        res = op.minimize(self.log_likelihood_call, self.p0, args=True, method='L-BFGS-B', bounds=bnds,\
                           options={'eps': 0.0001, 'disp':False, 'maxiter':20, 'ftol':1e-10,'maxcor':1000}) #'eps': 0.0001,
        if self.fit_sigma:
            if self.pattern_1_use:
                if self.pattern_2_use:
                    if self.pattern_3_use:
                        res['x'] *= self.p0_scale[0:7]
                    else:
                        res['x'] *= self.p0_scale[0:6]
                else:
                    res['x'] *= self.p0_scale[0:5]
            else:
                res['x'] *= self.p0_scale[0:4]
        else:
            if self.pattern_1_use:
                if self.pattern_2_use:
                    if self.pattern_3_use:
                        res['x'] *= self.p0_scale[0:6]
                    else:
                        res['x'] *= self.p0_scale[0:5]
                else:
                    res['x'] *= self.p0_scale[0:4]
            else:
                res['x'] *= self.p0_scale[0:3]
        self.res = res

# methods for calculating error
    def get_variance_from_hessian(self, x, enable_scale=False, func=''):
        x = np.array(x)
        x /= ft.p0_scale[0:len(x)] if enable_scale else np.ones(len(x))
        if func == 'likelihood':
            f = lambda x: ft.log_likelihood_call(x, enable_scale)
        elif func == 'chi_square':
            f = lambda x: ft.chi_square_call(x, enable_scale)
        else:
            raise ValueError('undefined function, should be likelihood or chi_square')
        H = nd.Hessian(f)  # ,step=1e-9)
        hh = H(x)
        hh_inv = np.linalg.inv(hh)
        variance = np.sqrt(np.diag(hh_inv))
        variance *= ft.p0_scale[0:5] if enable_scale else np.ones(len(x))
        return variance

    def get_location_errors(self, params, simulations, func='', first=None, last=None, delta=None):
        dx = params[0]
        dy = params[1]
        phi = params[2]
        events_rand = (params[3],)  # random
        events_per_sim = ()
        events_per_sim += (params[4],) if self.pattern_1_use else ()  # pattern 1
        events_per_sim += (params[5],) if self.pattern_2_use else ()  # pattern 2
        events_per_sim += (params[6],) if self.pattern_3_use else ()  # pattern 3
        # get patterns
        sims = ()
        sims += (simulations[0],) if self.pattern_1_use else ()
        sims += (simulations[1],) if self.pattern_2_use else ()
        sims += (simulations[2],) if self.pattern_3_use else ()
        print(events_rand, events_per_sim, sims)
        if first is None:
            first = 0
        if last is None:
            last = len(ft.lib.sim_list)
        if func == 'likelihood':
            if delta is None:
                delta = 0.5
            f = lambda s: -ft.log_likelihood(dx, dy, phi, events_rand, s, events_per_sim)
        elif func == 'chi_square':
            if delta is None:
                delta = 1.0
            f = lambda s: ft.chi_square(dx, dy, phi, events_rand, s, events_per_sim)
        else:
            raise ValueError('undefined function, should be likelihood or chi_square')
        estim_max = f(sims)
        crossings = []
        crossings_idx = []
        for i in range(len(sims)):
            estim = []
            temp_sims = np.array(sims)
            for x_sims in range(first, last):
                temp_sims[i] = x_sims
                estim.append(f(temp_sims))
                print(estim[x_sims]-(estim_max-delta))
            estim_diff = np.diff(estim)
            crossings_idx_temp = np.where(np.diff(np.sign(estim-(estim_max-delta))))[0]
            crossings_temp = crossings_idx - (estim-(estim_max-delta))[crossings_idx] / estim_diff[crossings_idx]
            crossings.append(crossings_temp)
            crossings_idx.append(crossings_idx_temp)
            print('crossings_idx - ', crossings_idx_temp)
            print('crossings - ', crossings_temp)
        return crossings, crossings_idx


if __name__ == "__main__":

    test_curve_fit = False
    test_chi2_min = True
    test_likelihood_max = False

    lib = lib2dl("/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_24Na/ue646g26.2dl")

    ft = fits(lib)

    # set a pattern to fit
    #x=np.arange(-1.79,1.8,0.01)
    #xmesh, ymesh = np.meshgrid(x,x)
    #xmesh, ymesh = create_detector_mesh(20, 20, 1.4, 300)
    #xmesh, ymesh = create_detector_mesh(50, 50, 0.5, 300)

    mm = MedipixMatrix(file_path='/home/eric/Desktop/jsontest.json')
    #patt = mm.matrixOriginal
    xmesh = mm.xmesh
    ymesh = mm.ymesh

    creator = PatternCreator(lib, xmesh, ymesh, (249-249,377-249))
    fractions_per_sim = np.array([0.65, 0.30, 0.05])
    total_events = 1e6
    # TODO fix montecarlo
    patt = creator.make_pattern(-1.08, 1.18, 179, fractions_per_sim, total_events, sigma=0.1, type='poisson')
    #patt = ma.masked_where(xmesh >=1.5,patt)
    patt = ma.array(data=patt, mask=mm.matrixOriginal.mask)

    plt.figure(0)
    plt.contourf(xmesh, ymesh, patt)#, np.arange(0, 3000, 100))
    plt.colorbar()
    plt.show(block=False)
    #plt.show()

    # set a fitting routine
    counts_ordofmag = 10**(int(math.log10(patt.sum())))
    ft.set_data_pattern(xmesh, ymesh, patt)
    ft.set_patterns_to_fit(249-249,377-249)
    ft.fit_sigma = True

    res = []
    if test_curve_fit:
        popt, pcov = ft.call_curve_fit()
        print('\noptimum values')
        print(popt)
        print('\nstandar dev')
        print(np.sqrt(np.diag(pcov)))
        print('\ncov matrix')
        print(pcov)

    if test_chi2_min:
        ft.set_scale_values(dx=1, dy=1, phi=1, total_cts=counts_ordofmag, sigma=1, f_p1=1)
        #ft.set_inicial_values(0.1, 0.1, 1, counts_ordofmag)
        ft.set_inicial_values(mm.center[0], mm.center[1], mm.angle, counts_ordofmag, sigma=0.1)
        ft.minimize_chi2()
        #var = ft.get_variance_from_hessian(res['x'],enable_scale=False,func='chi_square')
        #print('Calculating errors ...')
        #ft.print_variance(res['x'],var)
        print(ft.res)
        print('sigma in sim step units - ', ft.res['x'][4] / lib.xstep)
        # x = res['x'] * ft.p0_scale[0:5]
        # ft.set_scale_values()
        # # There is a warning because the hessian starts with a step too big, don't worry about it
        # H = nd.Hessian(ft.log_likelihood_call)#,step=1e-9)
        # hh = H(x)
        # print(hh)
        # print(np.linalg.inv(hh))
        # ft.set_scale_values(dx=1, dy=1, phi=10, f_rand=counts_ordofmag, f_p1=counts_ordofmag)
        # ft.print_results(res,hh)

    if test_likelihood_max:
        ft.set_scale_values(dx=1, dy=1, phi=1, total_cts=-1, f_p1=1, f_p2=1)
        #ft.set_inicial_values(0.1, 0.1, 1, -1)
        ft.set_inicial_values(mm.center[0], mm.center[1], mm.angle, -1, sigma=0.1)
        ft.maximize_likelyhood()
        print(ft.res)
        print('sigma in sim step units - ', ft.res['x'][4] / lib.xstep)
        print('Calculating errors ...')
        #var = ft.get_variance_from_hessian(res['x'],enable_scale=False,func='likelihood')
        #ft.print_variance(res['x'],var)
        #ft.get_location_errors(res['x'], (0,), last=300, func='likelihood')

    print('data points ', np.sum(~patt.mask))

    plt.figure(2)
    plt.contourf(xmesh, ymesh, ft.sim_pattern)
    plt.colorbar()

    plt.figure(3)
    if test_chi2_min:
        plt.contourf(xmesh, ymesh, ft.sim_pattern - patt)
    if test_likelihood_max:
        plt.contourf(xmesh, ymesh, ft.sim_pattern-patt/patt.sum())
    plt.colorbar()
    plt.show(block=True)

