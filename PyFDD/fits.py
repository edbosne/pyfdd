#!/usr/bin/env python3

'''
The fits object gets access to a lib2dl object and performs fits and statistical tests to data or MC simulation
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

from .lib2dl import lib2dl
from .patterncreator import PatternCreator, create_detector_mesh
from .MedipixMatrix import MedipixMatrix


import numpy as np
import numpy.ma as ma
import scipy.optimize as op
import scipy.stats as st
import math
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import importlib
iminuit_loader = importlib.find_loader('iminuit')
if iminuit_loader is not None:
    import iminuit




class fits:
    def __init__(self,lib):
        assert isinstance(lib,lib2dl)
        self.lib = lib
        # self.pattern_1_n = 0
        # self.pattern_1_use = True
        # self.pattern_2_n = None
        # self.pattern_2_use = False
        # self.pattern_3_n = None
        # self.pattern_3_use = False
        self.n_events = None
        self.n_events_set = False
        # self.fit_n_events = False
        self.XXmesh = None
        self.YYmesh = None
        self.data_pattern = None
        self.sim_pattern = None
        self.data_pattern_is_set = False
        #self.p0 = (None,)
        #self.p0_scale = np.ones((8))
        self.results = None
        self.std = None
        self.pattern_generator = None
        #self.sub_pixels = 1

        self.parameters_dict = None
        self._init_parameters_dict()
        self._parameters_order = ('dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3')
        self._pattern_keys = ('pattern_1', 'pattern_2', 'pattern_3')
        self._ml_fit_options = {'disp': False, 'maxiter': 30, 'maxfun': 300, 'ftol': 1e-8,'maxcor': 100}
        self._chi2_fit_options = {'disp': False, 'maxiter': 30, 'maxfun': 300, 'ftol': 1e-4, 'maxcor': 100}
        self._minuit_fit_options = {'tol': 0.1}
        self._minization_method = 'L-BFGS-B'
        self.verbose_graphics = False
        self.verbose_graphics_ax = None
        self.verbose_graphics_fg = None

    def _init_parameters_dict(self):
        parameter_template = \
            {'p0':None, 'value':None, 'use':False, 'std':None, 'scale':1, 'bounds':(None,None)}
        # parameters are, site 1 2 and 3,dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        # keys are 'pattern_1','pattern_2','pattern_3','sub_pixels','dx','dy','phi',
        # 'total_cts','sigma','f_p1','f_p2','f_p3'
        self.parameters_dict = {
            'pattern_1': parameter_template.copy(),  # site 1
            'pattern_2': parameter_template.copy(),  # site 2
            'pattern_3': parameter_template.copy(),  # site 3
            'sub_pixels': parameter_template.copy(),  # sub pixels for convolution
            'dx': parameter_template.copy(),  # delta x
            'dy': parameter_template.copy(),  # delta y
            'phi': parameter_template.copy(),  # rotation
            'total_cts': parameter_template.copy(), #total counts
            'sigma': parameter_template.copy(),  # sigma convolution
            'f_p1': parameter_template.copy(),  # fraction site 1
            'f_p2': parameter_template.copy(),  # fraction site 2
            'f_p3': parameter_template.copy(),  # fraction site 3
        }
        self.set_inicial_values()
        self.set_scale_values()
        self.set_bound_values()
        self.fix_parameters(dx=False, dy=False, phi=False, total_cts=False,
                            sigma=False, f_p1=False, f_p2=True, f_p3=True)

        self.parameters_dict['pattern_1']['value'] = 0
        self.parameters_dict['pattern_2']['value'] = 0
        self.parameters_dict['pattern_3']['value'] = 0
        self.parameters_dict['sub_pixels']['value'] = 1

    def set_inicial_values(self, dx=1., dy=1., phi=5., total_cts=1., sigma=0., f_p1=0.25, f_p2=0.25, f_p3=0.25):
        # parameter keys 'dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3'
        self.parameters_dict['dx']['p0'] = dx
        self.parameters_dict['dy']['p0'] = dy
        self.parameters_dict['phi']['p0'] = phi
        self.parameters_dict['total_cts']['p0'] = total_cts
        self.parameters_dict['sigma']['p0'] = sigma
        self.parameters_dict['f_p1']['p0'] = f_p1
        self.parameters_dict['f_p2']['p0'] = f_p2
        self.parameters_dict['f_p3']['p0'] = f_p3

    def set_scale_values(self, dx=1, dy=1, phi=1, total_cts=1, sigma=1, f_p1=1, f_p2=1, f_p3=1):
        # parameter keys 'dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3'
        self.parameters_dict['dx']['scale'] = dx
        self.parameters_dict['dy']['scale'] = dy
        self.parameters_dict['phi']['scale'] = phi
        self.parameters_dict['total_cts']['scale'] = total_cts
        self.parameters_dict['sigma']['scale'] = sigma
        self.parameters_dict['f_p1']['scale'] = f_p1
        self.parameters_dict['f_p2']['scale'] = f_p2
        self.parameters_dict['f_p3']['scale'] = f_p3

    def set_bound_values(self, dx=(-3, +3), dy=(-3, +3), phi=(None, None),
                         total_cts=(0, None), sigma=(0, None),
                         f_p1=(0, 1), f_p2=(0, 1), f_p3=(0, 1)):
        # parameter keys 'dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3'
        self.parameters_dict['dx']['bounds'] = dx
        self.parameters_dict['dy']['bounds'] = dy
        self.parameters_dict['phi']['bounds'] = phi
        self.parameters_dict['total_cts']['bounds'] = total_cts
        self.parameters_dict['sigma']['bounds'] = sigma
        self.parameters_dict['f_p1']['bounds'] = f_p1
        self.parameters_dict['f_p2']['bounds'] = f_p2
        self.parameters_dict['f_p3']['bounds'] = f_p3

    def fix_parameters(self, dx, dy, phi, total_cts, sigma,
                       f_p1, f_p2, f_p3):
        # parameter keys 'dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3'
        self.parameters_dict['dx']['use'] = not dx
        self.parameters_dict['dy']['use'] = not dy
        self.parameters_dict['phi']['use'] = not phi
        self.parameters_dict['total_cts']['use'] = not total_cts
        self.parameters_dict['sigma']['use'] = not sigma

        if self.parameters_dict['pattern_1']['use']:
            self.parameters_dict['f_p1']['use'] = not f_p1
        else:
            self.parameters_dict['f_p1']['use'] = False

        if self.parameters_dict['pattern_2']['use']:
            self.parameters_dict['f_p2']['use'] = not f_p2
        else:
            self.parameters_dict['f_p2']['use'] = False

        if self.parameters_dict['pattern_3']['use']:
            self.parameters_dict['f_p3']['use'] = not f_p3
        else:
            self.parameters_dict['f_p3']['use'] = False


    def set_optimization_profile(self,profile='default',min_method='L-BFGS-B'):
        # Using a coarse profile will lead to faster results and less optimized. tought sometimes it is also smoother
        # Using a fine profile can lead to rounding errors and jumping to other minima which causes artifacts
        # default eps is 1e-8 this, sometimes, is too small to correctly get the derivative of phi
        self._minization_method = min_method
        if min_method == 'L-BFGS-B':
            if profile == 'coarse':
                # if even with coarse the fit hangs consider other techniques for better fitting
                self._ml_fit_options =   {'disp':False, 'maxiter':10, 'maxfun':200, 'ftol':1e-7, 'maxcor':100, 'eps':1e-6}
                self._chi2_fit_options = {'disp':False, 'maxiter':10, 'maxfun':200, 'ftol':1e-6, 'maxcor':100, 'eps':1e-6}
            elif profile == 'default':
                self._ml_fit_options =   {'disp':False, 'maxiter':20, 'maxfun':200, 'ftol':1e-7, 'maxcor':100, 'eps':1e-6} #maxfun to 200 prevents memory problems
                self._chi2_fit_options = {'disp':False, 'maxiter':20, 'maxfun':300, 'ftol':1e-6, 'maxcor':100, 'eps':1e-6}
            elif profile == 'fine':
                # use default eps with fine
                self._ml_fit_options =   {'disp':False, 'maxiter':30, 'maxfun':300, 'ftol':1e-8, 'maxcor':100, 'eps':1e-7}
                self._chi2_fit_options = {'disp':False, 'maxiter':30, 'maxfun':600, 'ftol':1e-7, 'maxcor':100}
            else:
                raise ValueError('profile value should be set to: coarse, default or fine.')
        if min_method == 'minuit':
            if profile == 'coarse':
                # if even with coarse the fit hangs consider other techniques for better fitting
                self._minuit_fit_options = {'tol':100}
            elif profile == 'default':
                self._minuit_fit_options = {'tol': 1}
            elif profile == 'fine':
                # use default eps with fine
                self._minuit_fit_options = {'tol': 0.1}
            else:
                raise ValueError('profile value should be set to: coarse, default or fine.')

    def _get_p0_scale(self):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        p0_scale = ()
        for key in self._parameters_order:
            if self.parameters_dict[key]['use']:
                p0_scale += (self.parameters_dict[key]['scale'],)
        return np.array(p0_scale)

    def _get_p0(self):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        p0 = ()
        for key in self._parameters_order:
            if self.parameters_dict[key]['use']:
                temp_p0 = self.parameters_dict[key]['p0'] / self.parameters_dict[key]['scale']
                p0 += (temp_p0,)
        return np.array(p0)

    def set_data_pattern(self, XXmesh, YYmesh, pattern):
        self.XXmesh = XXmesh.copy()
        self.YYmesh = YYmesh.copy()
        self.data_pattern = pattern.copy()
        self.data_pattern_is_set = True
        self.n_events = self.data_pattern.sum()
        self.n_events_set = True

    def set_patterns_to_fit(self, p1_n=None, p2_n=None, p3_n=None):
        if p1_n is not None:
            self.parameters_dict['pattern_1']['value'] = p1_n
            self.parameters_dict['pattern_1']['use'] = True
            self.parameters_dict['f_p1']['use'] = True
        else:
            self.parameters_dict['pattern_1']['use'] = False
            self.parameters_dict['f_p1']['use'] = False

        if p2_n is not None:
            self.parameters_dict['pattern_2']['value'] = p2_n
            self.parameters_dict['pattern_2']['use'] = True
            self.parameters_dict['f_p2']['use'] = True
        else:
            self.parameters_dict['pattern_2']['use'] = False
            self.parameters_dict['f_p2']['use'] = False

        if p3_n is not None:
            self.parameters_dict['pattern_3']['value'] = p3_n
            self.parameters_dict['pattern_3']['use'] = True
            self.parameters_dict['f_p3']['use'] = True
        else:
            self.parameters_dict['pattern_3']['use'] = False
            self.parameters_dict['f_p3']['use'] = False

    def print_variance(self,x,var):
        # TODO remove
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

# methods for chi-square minimization
    def chi_square_fun(self, experimental_data, simlulation_data):
        # delta degrees of freedom
        # dx, dy, phi
        ddof = 3
        ddof += 1 if self.pattern_1_use else 0
        ddof += 1 if self.pattern_2_use else 0
        ddof += 1 if self.pattern_2_use else 0
        return st.chisquare(experimental_data, simlulation_data,ddof,axis=None)

    def chi_square(self, dx, dy, phi, total_events, fractions_sims, sigma=0):
        """
        Calculates the Pearson chi2 for the given conditions.
        :param dx: delta x in angles
        :param dy: delta y in angles
        :param phi: delta phi in anlges
        :param total_events: total number of events
        :param simulations: simulations id number
        :param fractions_sims: fractions of each simulated pattern
        :param sigma: sigma of the gaussian to convolute the pattern, smooting
        :return: Pearson's chi2
        """
        # set data pattern
        data_pattern = self.data_pattern
        fractions_sims = np.array(fractions_sims)
        rnd_events = np.array([1 - fractions_sims.sum()])
        # generate sim pattern
        gen = self.pattern_generator
        fractions = np.concatenate((rnd_events, fractions_sims))
        # mask out of range false means that points that are out of the range of simulations are not masked,
        # instead they are substituted by a very small number 1e-12
        sim_pattern = gen.make_pattern(dx, dy, phi, fractions, total_events, sigma=sigma, type='ideal',
                                       mask_out_of_range=False)
        self.sim_pattern = sim_pattern.copy()
        # chi2, pval = self.chi_square_fun(data_pattern,sim_pattern)
        chi2 = np.sum((data_pattern - sim_pattern) ** 2 / np.abs(sim_pattern))
        #print('chi2 - ', chi2)
        # print('p-value - ',pval)
        # =====
        if self.verbose_graphics:
            if self.verbose_graphics_ax is None or self.verbose_graphics_fg is None:
                fg = plt.figure()
                self.verbose_graphics_fg = fg
                ax = fg.add_subplot(111)
                self.verbose_graphics_ax = ax
                plt.show(block=False)
            plt.sca(self.verbose_graphics_ax)
            self.verbose_graphics_ax.clear()
            plt.ion()
            plt.contourf(self.XXmesh, self.YYmesh, sim_pattern) #(data_pattern-sim_pattern))
            self.verbose_graphics_fg.canvas.draw()
        # =====
        return chi2

    def chi_square_call(self, params, enable_scale=False):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        # print('params ', params)
        p0_scale = self._get_p0_scale() if enable_scale else np.ones(len(params))
        #print('p0_scale ', p0_scale)
        params_temp = ()
        di = 0
        for key in self._parameters_order:
            if self.parameters_dict[key]['use']:
                params_temp += (params[di] * p0_scale[di],)
                di += 1
            else:
                params_temp += (self.parameters_dict[key]['p0'],)
        # print('params_temp - ',params_temp)

        dx, dy, phi, total_cts, sigma, f_p1, f_p2, f_p3 = params_temp
        fractions_sims = ()
        fractions_sims += (f_p1,) if self.parameters_dict['pattern_1']['use'] else ()  # pattern 1
        fractions_sims += (f_p2,) if self.parameters_dict['pattern_2']['use'] else ()  # pattern 2
        fractions_sims += (f_p3,) if self.parameters_dict['pattern_3']['use'] else ()  # pattern 3
        #print('fractions_sims - ', fractions_sims)
        return self.chi_square(dx, dy, phi, total_cts, fractions_sims=fractions_sims, sigma=sigma)

    def minimize_chi2(self):
        self.minimize_cost_function(cost_func='chi2')

# methods for maximum likelihood
    def log_likelihood(self, dx, dy, phi, fractions_sims, sigma=0):
        """
        Calculates the Pearson chi2 for the given conditions.
        :param dx: delta x in angles
        :param dy: delta y in angles
        :param phi: delta phi in anlges
        :param simulations: simulations id number
        :param fractions_sims: fractions of each simulated pattern
        :param sigma: sigma of the gaussian to convolute the pattern, smooting
        :return: likelihood
        """
        # set data pattern
        data_pattern = self.data_pattern
        #if not len(simulations) == len(fractions_sims):
        #    raise ValueError("size o simulations is diferent than size o events")
        total_events = 1
        fractions_sims = np.array(fractions_sims)
        rnd_events = np.array([1 - fractions_sims.sum()])
        # generate sim pattern
        gen = self.pattern_generator
        # gen = PatternCreator(self.lib, self.XXmesh, self.YYmesh, simulations, mask=data_pattern.mask)
        fractions = np.concatenate((rnd_events, fractions_sims))
        sim_pattern = gen.make_pattern(dx, dy, phi, fractions, total_events, sigma=sigma, type='ideal',
                                       mask_out_of_range=False)
        self.sim_pattern = sim_pattern.copy()
        # log likelihood
        ll = np.sum(data_pattern * np.log(sim_pattern))
        # extended log likelihood - no need to fit events
        #ll = -np.sum(events_per_sim) + np.sum(data_pattern * np.log(sim_pattern))
        #print('likelihood - ', ll)
        # =====
        if self.verbose_graphics:
            if self.verbose_graphics_ax is None or self.verbose_graphics_fg is None:
                fg = plt.figure()
                self.verbose_graphics_fg = fg
                ax = fg.add_subplot(111)
                self.verbose_graphics_ax = ax
                plt.show(block=False)
            plt.sca(self.verbose_graphics_ax)
            plt.ion()
            plt.contourf(self.XXmesh, self.YYmesh, sim_pattern)  # (data_pattern-sim_pattern))
            self.verbose_graphics_fg.canvas.draw()
        # =====
        return -ll


    def log_likelihood_call(self, params, enable_scale=False):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        #print('params ', params)
        p0_scale = self._get_p0_scale() if enable_scale else np.ones(len(params))
        # print('p0_scale ', p0_scale)
        params_temp = ()
        di = 0
        for key in self._parameters_order:
            if self.parameters_dict[key]['use']:
                params_temp += (params[di] * p0_scale[di],)
                di += 1
            else:
                params_temp += (self.parameters_dict[key]['p0'],)
        #print('params_temp - ',params_temp)

        dx, dy, phi, total_cts, sigma, f_p1, f_p2, f_p3 = params_temp
        fractions_sims = ()
        fractions_sims += (f_p1,) if self.parameters_dict['pattern_1']['use'] else () # pattern 1
        fractions_sims += (f_p2,) if self.parameters_dict['pattern_2']['use'] else () # pattern 2
        fractions_sims += (f_p3,) if self.parameters_dict['pattern_3']['use'] else () # pattern 3
        #print('fractions_sims - ', fractions_sims)
        return self.log_likelihood(dx, dy, phi, fractions_sims, sigma=sigma)

    def maximize_likelyhood(self):
        self.minimize_cost_function(cost_func='ml')

    def minimize_cost_function(self, cost_func='chi2'):

        if not cost_func in ('ml', 'chi2'):
            raise ValueError('cost function should be \'chi2\' or \'ml\'')

        if cost_func == 'ml':
            # total counts is not used in maximum likelyhood
            self.parameters_dict['total_cts']['use'] = False

        # order of params is dx,dy,phi,sigma,f_p1,f_p2,f_p3
        p0 = self._get_p0()
        # print('p0 - ', p0)

        # Parameter bounds
        bnds = ()
        for key in self._parameters_order:
            if self.parameters_dict[key]['use']:
                bnds += (self.parameters_dict[key]['bounds'],)
        # print('bnds - ', bnds)

        # get patterns
        simulations = ()
        for key in self._pattern_keys:
            if self.parameters_dict[key]['use']:
                simulations += (self.parameters_dict[key]['value'],)
        # print('simulations - ', simulations)

        # generate sim pattern
        self.pattern_generator = PatternCreator(self.lib, self.XXmesh, self.YYmesh, simulations,
                                                mask=self.data_pattern.mask,
                                                sub_pixels=self.parameters_dict['sub_pixels']['value'])

        # defining cost function and get options
        function = None
        all_options = {}
        if cost_func == 'chi2':
            function = self.chi_square_call
            all_options = self._chi2_fit_options
        elif cost_func == 'ml':
            function = self.log_likelihood_call
            all_options = self._ml_fit_options

        # select method
        if self._minization_method == 'minuit':
            minuit = self._create_minuit(cost_func)
            res = minuit.migrad()
            print('migrad res', res)
            self.results = res

            for param_res in res:
                if 'name' not in param_res.keys():
                    continue
                key = param_res['name']
                # scale is inactive when using minuit
                self.parameters_dict[key]['value'] = param_res['value']
                self.parameters_dict[key]['std'] = param_res['error']
        else:
            res = op.minimize(function, p0, args=True, method=self._minization_method, bounds=bnds, \
                              options=all_options)  # 'eps': 0.0001, L-BFGS-B
            # minimization with cobyla also seems to be a good option with {'rhobeg':1e-1/1e-2} . but it is unconstrained
            di = 0
            for key in self._parameters_order:
                if self.parameters_dict[key]['use']:
                    res['x'][di] *= self.parameters_dict[key]['scale']
                    self.parameters_dict[key]['value'] = res['x'][di]
                    di += 1
                else:
                    self.parameters_dict[key]['value'] = self.parameters_dict[key]['p0']

            self.results = res

    def _create_minuit(self, cost_func='chi2', options={}):
        # parameters_order is ('dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3')
        arguments = {}

        # value for calculating a 1 sigma error
        if cost_func == 'chi2':
            arguments['errordef'] = 1
        elif cost_func == 'ml':
            arguments['errordef'] = 0.5
        for key in self._parameters_order:
            if cost_func == 'ml' and key == 'total_cts':
                continue

            # starting value
            arguments[key] = self.parameters_dict[key]['p0']

            # fix if not in use
            if self.parameters_dict[key]['use']:
                arguments['fix_' + key] = False
            else:
                arguments['fix_' + key] = True
                continue
            arguments['error_' + key] = 1e-3 * self.parameters_dict[key]['scale']

            # bounds
            arguments['limit_' + key] = self.parameters_dict[key]['bounds']
        print('arguments\n', arguments)
        minuit =  None
        if cost_func == 'chi2':
            minuit = iminuit.Minuit(self.chi_square_call_migrad, **arguments)
        elif cost_func == 'ml':
            minuit = iminuit.Minuit(self.log_likelihood_call_migrad, **arguments)
        minuit.tol = self._minuit_fit_options['tol']
        print('tol', minuit.tol)
        return minuit


    def log_likelihood_call_migrad(self, dx, dy, phi, sigma, f_p1, f_p2, f_p3):
        fractions_sims = ()
        fractions_sims += (f_p1,) if self.parameters_dict['pattern_1']['use'] else ()  # pattern 1
        fractions_sims += (f_p2,) if self.parameters_dict['pattern_2']['use'] else ()  # pattern 2
        fractions_sims += (f_p3,) if self.parameters_dict['pattern_3']['use'] else ()  # pattern 3
        # print('fractions_sims - ', fractions_sims)
        return self.log_likelihood(dx, dy, phi, fractions_sims, sigma=sigma)


    def chi_square_call_migrad(self, dx, dy, phi, total_cts, sigma, f_p1, f_p2, f_p3):
        fractions_sims = ()
        fractions_sims += (f_p1,) if self.parameters_dict['pattern_1']['use'] else ()  # pattern 1
        fractions_sims += (f_p2,) if self.parameters_dict['pattern_2']['use'] else ()  # pattern 2
        fractions_sims += (f_p3,) if self.parameters_dict['pattern_3']['use'] else ()  # pattern 3
        # print('fractions_sims - ', fractions_sims)
        return self.chi_square(dx, dy, phi, total_cts, fractions_sims, sigma=sigma)


# methods for calculating error
    def get_std_from_hessian(self, x, enable_scale=False, func=''):
        x = np.array(x)
        x /= self._get_p0_scale() if enable_scale else np.ones(len(x))
        if func == 'likelihood':
            f = lambda xx: self.log_likelihood_call(xx, enable_scale)
        elif func == 'chi_square':
            f = lambda xx: self.chi_square_call(xx, enable_scale)
        else:
            raise ValueError('undefined function, should be likelihood or chi_square')
        H = nd.Hessian(f)  # ,step=1e-9)
        hh = H(x)
        if func == 'likelihood':
            hh_inv = np.linalg.inv(hh)
        elif func == 'chi_square':
            hh_inv = np.linalg.inv(0.5*hh)
        else:
            raise ValueError('undefined function, should be likelihood or chi_square')
        std = np.sqrt(np.diag(hh_inv))
        std *= self._get_p0_scale() if enable_scale else np.ones(len(x))
        self.std = std
        di = 0
        for key in self._parameters_order:
            if self.parameters_dict[key]['use']:
                self.parameters_dict[key]['std'] = std[di]
                di += 1
        return std

    def get_location_errors(self, params, simulations, func='', first=None, last=None, delta=None):
        dx = params[0]
        dy = params[1]
        phi = params[2]
        events_rand = (params[3],)  # random
        events_per_sim = ()
        events_per_sim += (params[4],) if self.parameters_dict['pattern_1']['use'] else ()  # pattern 1
        events_per_sim += (params[5],) if self.parameters_dict['pattern_2']['use'] else ()  # pattern 2
        events_per_sim += (params[6],) if self.parameters_dict['pattern_3']['use'] else ()  # pattern 3
        # get patterns
        sims = ()
        sims += (simulations[0],) if self.parameters_dict['pattern_1']['use'] else ()
        sims += (simulations[1],) if self.parameters_dict['pattern_2']['use'] else ()
        sims += (simulations[2],) if self.parameters_dict['pattern_3']['use'] else ()
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


