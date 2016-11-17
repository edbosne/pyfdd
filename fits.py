#!/usr/bin/env python3

'''
The fits object gets access to a lib2dl object and performs fits and statistical tests to data or MC simulation
'''
import patterncreator

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

from lib2dl import lib2dl
from patterncreator import *

import numpy as np
import scipy.optimize as op
import scipy.stats as st
import matplotlib.pyplot as plt
import math
import numdifftools as nd


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
        self.data_pattern_is_set = False
        self.p0 = (None,)
        self.p0_scale = np.ones((7))

    def set_data_pattern(self,XXmesh,YYmesh,pattern,mask=None):
        self.XXmesh = XXmesh
        self.YYmesh = YYmesh
        self.data_pattern = pattern
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

    def set_inicial_values(self,dx=1,dy=1,phi=5,f_rand=0.5,f_p1=0.5,f_p2=0.5,f_p3=0.5):
        p0 = (dx/self.p0_scale[0],)
        p0 += (dy/self.p0_scale[1],)
        p0 += (phi/self.p0_scale[2],)
        p0 += (f_rand/self.p0_scale[3],)
        p0 += (f_p1/self.p0_scale[4],) if self.pattern_1_use else ()
        p0 += (f_p2/self.p0_scale[5],) if self.pattern_2_use else ()
        p0 += (f_p3/self.p0_scale[6],) if self.pattern_3_use else ()
        self.p0 = np.array(p0)

    def set_scale_values(self, dx=1, dy=1, phi=1, f_rand=1, f_p1=1, f_p2=1, f_p3=1):
        p0_scale = (dx,)
        p0_scale += (dy,)
        p0_scale += (phi,)
        p0_scale += (f_rand,)
        p0_scale += (f_p1,)
        p0_scale += (f_p2,)
        p0_scale += (f_p3,)
        self.p0_scale = np.array(p0_scale)

    def print_results(self,results,hess):
        assert isinstance(results,op.OptimizeResult)
        scale = np.diag(1/self.p0_scale[0:5])
        hess_inv = np.linalg.inv(hess) #results['hess_inv'].todense()
        #hess = np.linalg.inv(hess_inv)
        #hess_inv = np.linalg.inv(scale.dot(hess).dot(scale))
        params = results['x']
        print(hess_inv,params,self.p0_scale)
        dx = params[0] * self.p0_scale[0]
        d_dx = np.sqrt(hess_inv[0,0]) * self.p0_scale[0]
        dy = params[1] * self.p0_scale[1]
        d_dy = np.sqrt(hess_inv[1,1]) * self.p0_scale[1]
        phi = params[2] * self.p0_scale[2]
        d_phi = np.sqrt(hess_inv[2,2])
        N_rand = params[3] * self.p0_scale[3]
        d_N_rand = np.sqrt(hess_inv[3,3])
        N_p1 = params[4] * self.p0_scale[4] if self.pattern_1_use else 0
        d_N_p1 = np.sqrt(hess_inv[4,4]) if self.pattern_1_use else 0
        N_p2 = params[5] * self.p0_scale[5] * self.p0_scale[5] if self.pattern_2_use else 0
        d_N_p2 = np.sqrt(hess_inv[5,5]) * self.p0_scale[5] if self.pattern_2_use else 0
        N_p3 = params[6] * self.p0_scale[6] if self.pattern_3_use else 0
        d_N_p3 = np.sqrt(hess_inv[6,6]) * self.p0_scale[6] if self.pattern_3_use else 0

        total_f = N_rand + N_p1 + N_p2 + N_p3
        f_rand = N_rand/total_f
        f_1 = N_p1/total_f
        f_2 = N_p2/total_f
        f_3 = N_p3/total_f
        print('rand',N_rand, d_N_rand)
        print('p1', N_p1, d_N_p1)
        d_f_rand = np.abs(d_N_rand / total_f \
                   - N_rand * (d_N_rand /total_f**2 + d_N_p1 /total_f**2 + d_N_p2 /total_f**2 + d_N_p3 /total_f**2))
        d_f_1 = np.abs(d_N_p1 / total_f \
                   - N_p1   * (d_N_rand /total_f**2 + d_N_p1 /total_f**2 + d_N_p2 /total_f**2 + d_N_p3 /total_f**2))
        d_f_2 = np.abs(d_N_p2 / total_f \
                   - N_p2 * (d_N_rand /total_f**2 + d_N_p1 /total_f**2 + d_N_p2 /total_f**2 + d_N_p3 /total_f**2))
        d_f_3 = np.abs(d_N_p3 / total_f \
                   - N_p3 * (d_N_rand /total_f**2 + d_N_p1 /total_f**2 + d_N_p2 /total_f**2 + d_N_p3 /total_f**2))

        res = {'dx':dx, 'd_dx':d_dx,
               'dy': dy, 'd_dy': d_dy,
               'phi':phi, 'd_phi':d_phi,
               'f_rand':f_rand,'d_f_rand':d_f_rand,
               'f_1':f_1, 'd_f_1':d_f_1,
               'f_2': f_2, 'd_f_2': d_f_2,
               'f_3': f_3, 'd_f_3': d_f_3}


        print(('dx     = {dx:.4f} +- {d_dx:.4f}\n'+
              'dy     = {dy:.4f} +- {d_dy:.4f}\n'+
              'phi    = {phi:.4f} +- {d_phi:.4f}\n'+
              'f_rand = {f_rand:.4f} +- {d_f_rand:.4f}\n'+
              'f_1    = {f_1:.4f} +- {d_f_1:.4f}\n'+
              'f_2    = {f_2:.4f} +- {d_f_2:.4f}\n'+
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

    def chi_square_call(self, params):
        print(params)
        dx = params[0] * self.p0_scale[0]
        dy = params[1] * self.p0_scale[1]
        phi = params[2] * self.p0_scale[2]
        events_per_sim  = (params[3] * self.p0_scale[3],) # random
        events_per_sim += (params[4] * self.p0_scale[4],) if self.pattern_1_use else () # pattern 1
        events_per_sim += (params[5] * self.p0_scale[5],) if self.pattern_2_use else () # pattern 2
        events_per_sim += (params[6] * self.p0_scale[6],) if self.pattern_3_use else () # pattern 3
        # get patterns
        simulations  = (self.pattern_1_n,) if self.pattern_1_use else ()
        simulations += (self.pattern_2_n,) if self.pattern_2_use else ()
        simulations += (self.pattern_3_n,) if self.pattern_3_use else ()
        # generate sim pattern
        gen = PatternCreator(self.lib, self.XXmesh, self.YYmesh, simulations)
        sim_pattern = gen.make_pattern(dx, dy, phi, events_per_sim, 'ideal')
        # set data pattern
        data_pattern = self.data_pattern
        #chi2, pval = self.chi_square_fun(data_pattern,sim_pattern)
        chi2 = np.sum((data_pattern - sim_pattern)**2 / np.abs(sim_pattern))
        print('chi2 - ',chi2)
        #print('p-value - ',pval)
        # =====
        # fg = plt.figure(1)
        # ax = fg.add_subplot(111)
        # plt.ion()
        # cont = None
        # plt.contourf(self.XXmesh, self.YYmesh, sim_pattern)
        # fg.canvas.draw()
        # TODO fix bug in lib.XXmesh shape
        # fg = plt.figure(2)
        # ax = fg.add_subplot(111)
        # plt.ion()
        # cont = None
        # plt.contourf(lib.XXmesh, lib.YYmesh, lib.pattern_current)
        # fg.canvas.draw()
        # fg = plt.figure(3)
        # ax = fg.add_subplot(111)
        # plt.ion()
        # cont = None
        # plt.contourf(lib.XXmesh_original, lib.YYmesh_original, lib.pattern_original)
        # fg.canvas.draw()
        # plt.show(block=False)
        # =====
        return chi2

    def minimize_chi2(self):
        p0 = self.p0
        print('p0 - ', p0)

        bnds = ((-0.5,+0.5), (-0.5,+0.5), (None,None), (0, None), (0, None))

        return op.minimize(self.chi_square_call,p0,method='L-BFGS-B', bounds=bnds,\
                           options={'disp':True, 'maxiter':20, 'ftol':1e-7,'maxcor':1000}) #'eps':0.001,
        #return op.minimize(self.chi_square_call,p0,method='Powell',options={'direc':[0.1,0.1,2,-0.1,-0.1]})

# methods for maximum likelihood
    # TODO independent log likelyhood

    def log_likelihood_call(self, params):
        # TODO enable scale option
        print(params)
        dx = params[0] * self.p0_scale[0]
        dy = params[1] * self.p0_scale[1]
        phi = params[2] * self.p0_scale[2]
        events_per_sim  = (params[3] * self.p0_scale[3],) # random
        events_per_sim += (params[4] * self.p0_scale[4],) if self.pattern_1_use else () # pattern 1
        events_per_sim += (params[5] * self.p0_scale[5],) if self.pattern_2_use else () # pattern 2
        events_per_sim += (params[6] * self.p0_scale[6],) if self.pattern_3_use else () # pattern 3
        # get patterns
        simulations  = (self.pattern_1_n,) if self.pattern_1_use else ()
        simulations += (self.pattern_2_n,) if self.pattern_2_use else ()
        simulations += (self.pattern_3_n,) if self.pattern_3_use else ()
        # generate sim pattern
        gen = PatternCreator(self.lib, self.XXmesh, self.YYmesh, simulations)
        sim_pattern = gen.make_pattern(dx, dy, phi, events_per_sim, 'ideal')
        # set data pattern
        data_pattern = self.data_pattern
        # extended log likelihood
        ll = -np.sum(events_per_sim) + np.sum(data_pattern*np.log(sim_pattern))
        print('likelihood - ',ll)
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

    def maximize_likelyhood(self):
        # TODO raise error is parameters diferent from patterns
        p0 = self.p0
        print('p0 - ', p0)

        bnds = ((-0.5,+0.5), (-0.5,+0.5), (None,None), (0, None), (0, None))
        # TODO return unscaled results
        return op.minimize(self.log_likelihood_call, p0, method='L-BFGS-B', bounds=bnds,\
                           options={'eps': 0.001, 'disp':True, 'maxiter':20, 'ftol':1e-6,'maxcor':1000}) #'eps': 0.00001,

        #return op.minimize(self.log_likelihood_call, p0, method='TNC', bounds=bnds, \
        #                 options={'disp': True, 'maxiter': 20, 'ftol': 1e-6,
         #                           'scale':self.p0_scale_1[0:5]})  # 'eps': 0.00001,

if __name__ == "__main__":

    test_curve_fit = False
    test_chi2_min = True
    test_likelihood_max = False

    lib = lib2dl("/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_89Sr/ue567g54.2dl")

    ft = fits(lib)

    # set a pattern to fit
    x=np.arange(-1.79,1.8,0.01)
    #xmesh, ymesh = np.meshgrid(x,x)
    #xmesh, ymesh = create_detector_mesh(22, 22, 1.4, 300)
    xmesh, ymesh = create_detector_mesh(50, 50, 0.5, 300)

    creator = PatternCreator(lib, xmesh, ymesh, 0)
    events_per_sim = np.array([0.5, 0.5]) * 1e6
    patt = creator.make_pattern(0.2, -0.2, 5, events_per_sim, 'montecarlo')

    plt.figure(0)
    plt.contourf(xmesh, ymesh, patt)#, np.arange(0, 3000, 100))
    plt.colorbar()
    plt.show(block=False)

    # set a fitting routine
    counts_ordofmag = 10**(int(math.log10(patt.sum())))
    ft.set_data_pattern(xmesh,ymesh,patt)
    ft.set_patterns_to_fit(0)
    ft.set_scale_values(dx=1, dy=1, phi=10, f_rand=counts_ordofmag, f_p1=counts_ordofmag)
    ft.set_inicial_values(0.0,0.1,0,counts_ordofmag/3,counts_ordofmag/3)

    if test_curve_fit:
        popt, pcov = ft.call_curve_fit()
        print('\noptimum values')
        print(popt)
        print('\nstandar dev')
        print(np.sqrt(np.diag(pcov)))
        print('\ncov matrix')
        print(pcov)

    if test_chi2_min:
        res = ft.minimize_chi2()
        print(res)
        x = res['x'] * ft.p0_scale[0:5]
        ft.set_scale_values()
        # There is a warning because the hessian starts with a step too big, don't worry about it
        H = nd.Hessian(ft.log_likelihood_call)#,step=1e-9)
        hh = H(x)
        print(hh)
        print(np.linalg.inv(hh))
        ft.set_scale_values(dx=1, dy=1, phi=10, f_rand=counts_ordofmag, f_p1=counts_ordofmag)
        ft.print_results(res,hh)

    if test_likelihood_max:
        res = ft.maximize_likelyhood()
        print(res)
        x = res['x'] * ft.p0_scale[0:5]
        ft.set_scale_values()
        # There is a warning because the hessian starts with a step too big, don't worry about it
        H = nd.Hessian(ft.log_likelihood_call)#,step=1e-9)
        hh = H(x)
        print(hh)
        print(np.linalg.inv(hh))
        ft.set_scale_values(dx=1, dy=1, phi=10, f_rand=counts_ordofmag, f_p1=counts_ordofmag)
        ft.print_results(res, hh)

