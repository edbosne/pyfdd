#!/usr/bin/env python3

'''
The Fit object gets access to a Lib2dl object and performs Fit and statistical tests to data or MC simulation
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

from .lib2dl import Lib2dl
from .patterncreator import PatternCreator, create_detector_mesh
from .datapattern import DataPattern


import numpy as np
import numpy.ma as ma
import scipy.optimize as op
import scipy.stats as st
import math
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import collections
import warnings



class Fit:
    def __init__(self, lib, sites, verbose_graphics=False):
        '''
        Init method for class fit
        :param lib: Lib2dl library
        :param sites: indexes of sites to include in the fit
        :param verbose_graphics: plot graphics while fitting
        '''

        if not isinstance(lib, Lib2dl):
            raise ValueError('lib is not an instance of Lib2dl')

        if not isinstance(sites, collections.Sequence):
            if isinstance(sites, (int, np.integer)):
                sites = (sites,)
            else:
                raise ValueError('sites needs to be an int or a sequence of ints')
        for s in sites:
            if not isinstance(s, (int, np.integer)):
                raise ValueError('sites needs to be an int or a sequence of ints')

        # library file and sites idexes
        self._lib = lib
        self._sites_idx = sites
        self._n_sites = len(sites)

        # data and simulation pattern variables
        self.XXmesh = None
        self.YYmesh = None
        self.data_pattern = None
        self.sim_pattern = None
        self.data_pattern_is_set = False

        # results and error bars
        self.results = None
        self.std = None
        self.pattern_generator = None

        # fit parameters and settings
        self._parameters_dict = None
        self._parameters_order = None
        self._pattern_keys = None
        self._init_parameters_variables()
        self.previous_cost_value = None


        # minimization default options
        self._fit_options = {'disp': False, 'maxiter': 30, 'maxfun': 300, 'ftol': 1e-8, 'maxcor': 100}
        self._minimization_method = 'L-BFGS-B'

        # visualisation
        self.verbose_graphics = verbose_graphics
        self.verbose_graphics_ax = None
        self.verbose_graphics_fg = None

    def _init_parameters_variables(self):
        parameter_template = \
            {'p0':None, 'value':None, 'use':True, 'std':None, 'scale':1, 'bounds':(None,None)}
        # parameters are, site 1 2 and 3,dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        # keys are 'sub_pixels','dx','dy','phi',
        # 'total_cts','sigma','pattern_1', 'f_p1', 'pattern_2', 'f_p2', 'pattern_3', 'f_p3'

        # base parameters_dict (without sites)
        self._parameters_dict = {
            'sub_pixels': parameter_template.copy(),  # sub pixels for convolution
            'dx': parameter_template.copy(),  # delta x
            'dy': parameter_template.copy(),  # delta y
            'phi': parameter_template.copy(),  # rotation
            'total_cts': parameter_template.copy(), #total counts
            'sigma': parameter_template.copy(),  # sigma convolution
        }
        # adding site variables
        for i in np.arange(1, 1+self._n_sites):
            pattern = 'pattern_' + str(i)
            fraction = 'f_p' + str(i)
            new_dict = {pattern: parameter_template.copy(),  # site idx i
                        fraction: parameter_template.copy()}  # fraction site i
            self._parameters_dict = {**self._parameters_dict, **new_dict}

        # order of parameters
        self._parameters_order = ('dx', 'dy', 'phi', 'total_cts', 'sigma')
        for i in np.arange(1, 1 + self._n_sites):
            fraction = 'f_p' + str(i)
            self._parameters_order += (fraction,)

        # pattern keys
        self._pattern_keys = ()
        for i in np.arange(1, 1+self._n_sites):
            pattern = 'pattern_' + str(i)
            self._pattern_keys += (pattern,)

        # starting values
        self._set_patterns_to_fit()
        self._parameters_dict['sub_pixels']['value'] = 1

        self.set_inicial_values()
        self.set_scale_values()
        self.set_bound_values()

    def set_inicial_values(self, dx=1., dy=1., phi=5., total_cts=1., sigma=0., **kwargs):#f_p1=0.25, f_p2=0.25, f_p3=0.25):
        # parameter keys 'dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3'
        self._parameters_dict['dx']['p0'] = dx
        self._parameters_dict['dy']['p0'] = dy
        self._parameters_dict['phi']['p0'] = phi
        self._parameters_dict['total_cts']['p0'] = total_cts
        self._parameters_dict['sigma']['p0'] = sigma
        for i in np.arange(1, 1 + self._n_sites):
            fraction = 'f_p' + str(i)
            self._parameters_dict[fraction]['p0'] = kwargs.pop(fraction, 0.25)
        if kwargs:
            raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))

    def set_scale_values(self, dx=1, dy=1, phi=1, total_cts=1, sigma=1, **kwargs):#f_p1=1, f_p2=1, f_p3=1):
        # parameter keys 'dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3'
        self._parameters_dict['dx']['scale'] = dx
        self._parameters_dict['dy']['scale'] = dy
        self._parameters_dict['phi']['scale'] = phi
        self._parameters_dict['total_cts']['scale'] = total_cts
        self._parameters_dict['sigma']['scale'] = sigma
        for i in np.arange(1, 1 + self._n_sites):
            fraction = 'f_p' + str(i)
            self._parameters_dict[fraction]['scale'] = kwargs.pop(fraction, 1)
        if kwargs:
            raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))

    def set_bound_values(self, dx=(-3, +3), dy=(-3, +3), phi=(None, None),
                         total_cts=(1, None), sigma=(0.01, None), **kwargs):#
                         #f_p1=(0, 1), f_p2=(0, 1), f_p3=(0, 1)):
        # parameter keys 'dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3'
        self._parameters_dict['dx']['bounds'] = dx
        self._parameters_dict['dy']['bounds'] = dy
        self._parameters_dict['phi']['bounds'] = phi
        self._parameters_dict['total_cts']['bounds'] = total_cts
        self._parameters_dict['sigma']['bounds'] = sigma
        for i in np.arange(1, 1 + self._n_sites):
            fraction = 'f_p' + str(i)
            self._parameters_dict[fraction]['bounds'] = kwargs.pop(fraction, (0, 1))
        if kwargs:
            raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))

    def fix_parameters(self, *, dx, dy, phi, total_cts, sigma, **kwargs):#
                       #f_p1, f_p2, f_p3):
        # parameter keys 'dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3'
        self._parameters_dict['dx']['use'] = not dx
        self._parameters_dict['dy']['use'] = not dy
        self._parameters_dict['phi']['use'] = not phi
        self._parameters_dict['total_cts']['use'] = not total_cts
        self._parameters_dict['sigma']['use'] = not sigma
        for i in np.arange(1, 1 + self._n_sites):
            fraction = 'f_p' + str(i)
            self._parameters_dict[fraction]['use'] = not kwargs.pop(fraction, False)
        if kwargs:
            raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))

    def _fix_duplicated_sites(self):
        for n, idx in enumerate(self._sites_idx):
            if idx in self._sites_idx[:n]:
                fraction = 'f_p' + str(n+1)
                self._parameters_dict[fraction]['use'] = False
                self._parameters_dict[fraction]['p0'] = 0

    def set_fit_options(self, options):

        if not isinstance(options, dict):
            raise ValueError('options must be of type dict')

        for key in options.keys():
            self._fit_options[key] = options[key]

    def set_sub_pixels(self, sub_pixels):
        self._parameters_dict['sub_pixels']['value'] = sub_pixels

    def _get_p0_scale(self):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        p0_scale = ()
        for key in self._parameters_order:
            if self._parameters_dict[key]['use']:
                p0_scale += (self._parameters_dict[key]['scale'],)
        return np.array(p0_scale)

    def _get_p0(self):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        # only parameters that are changed in the fit are given.
        p0 = ()
        for key in self._parameters_order:
            if self._parameters_dict[key]['use']:
                temp_p0 = self._parameters_dict[key]['p0'] / self._parameters_dict[key]['scale']
                p0 += (temp_p0,)
        return np.array(p0)

    def _get_bounds(self):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        bnds = ()

        for key in self._parameters_order:
            if self._parameters_dict[key]['use']:
                if self._parameters_dict[key]['bounds'][0] is not None:
                    temp_0 = self._parameters_dict[key]['bounds'][0] / self._parameters_dict[key]['scale']
                else:
                    temp_0 = None
                if self._parameters_dict[key]['bounds'][1] is not None:
                    temp_1 = self._parameters_dict[key]['bounds'][1] / self._parameters_dict[key]['scale']
                else:
                    temp_1 = None
                bnds += ((temp_0,temp_1),)
        #print('bnds - ', bnds)
        return bnds

    def set_data_pattern(self, XXmesh, YYmesh, pattern):
        self.XXmesh = XXmesh.copy()
        self.YYmesh = YYmesh.copy()
        self.data_pattern = pattern.copy()
        self.data_pattern_is_set = True

    def _set_patterns_to_fit(self):
        for i in np.arange(0, self._n_sites):
            k = self._pattern_keys[i]
            self._parameters_dict[k]['value'] = self._sites_idx[i]
            self._parameters_dict[k]['use'] = True
            self._parameters_dict['f_p'+str(i+1)]['use'] = True

# Fit methods
# methods for chi-square minimization
    def get_dof(self):
        """
        Returns the number of degrees of freedom
        :return: number of degrees of freedom
        """

        # getting the number of data points
        if isinstance(self.data_pattern, np.ma.MaskedArray):
            n_pixels = (~self.data_pattern.mask).sum()
        elif isinstance(self.data_pattern, np.Array):
            n_pixels = self.data_pattern.size
        else:
            raise ValueError('The data pattern is not correctly set')

        # getting the number of fit parameters
        n_param = 0
        for key in self._parameters_order:
            if self._parameters_dict[key]['use']:
                n_param += 1

        return n_pixels - n_param

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
        # generate sim pattern
        gen = self.pattern_generator
        # mask out of range false means that points that are out of the range of simulations are not masked,
        # instead they are substituted by a very small number 1e-12
        sim_pattern = gen.make_pattern(dx, dy, phi, fractions_sims, total_events, sigma=sigma, type='ideal')
        self.sim_pattern = sim_pattern.copy()
        #chi2 = np.sum((data_pattern - sim_pattern) ** 2 / np.abs(sim_pattern))
        chi2 = np.sum((data_pattern - sim_pattern)**2 / sim_pattern)
        #print('chi2 - {:0.12f}'.format(chi2))
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
            self.verbose_graphics_ax.set_aspect('equal')
            self.verbose_graphics_fg.canvas.draw()
        # =====
        return chi2

    def chi_square_call(self, params, enable_scale=False):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        #print('params ', params)
        p0_scale = self._get_p0_scale() if enable_scale else np.ones(len(params))
        #print('p0_scale ', p0_scale, enable_scale)
        params_temp = ()
        di = 0
        for key in self._parameters_order:
            if self._parameters_dict[key]['use']:
                params_temp += (params[di] * p0_scale[di],)
                di += 1
            else:
                params_temp += (self._parameters_dict[key]['p0'],)
        # print('params_temp - ',params_temp)

        dx, dy, phi, total_cts, sigma = params_temp[0:5]

        fractions_sims = ()
        for i in range(self._n_sites):
            fractions_sims += (params_temp[5 + i],)  # fractions f_p1, f_p2, f_p3,...
        #print('fractions_sims - ', fractions_sims, self._n_sites)

        chi2 = self.chi_square(dx, dy, phi, total_cts, fractions_sims=fractions_sims, sigma=sigma)

        # This is sort of a hack to avoid that after a very high chi2
        # the next point is not exactly the same as the previous.
        # This happens as if after a step in the direction of -gradient
        # The function value increases an intermediate step is chosen
        # at a position weigthed by the values at each extreme.
        # This allows the pattern to adjust better to the edges of the
        # experimental vs theoretical range.
        if self.previous_cost_value is None:
            self.previous_cost_value = chi2
        elif chi2 > self.previous_cost_value * 1e12:
            # limit the increase to 12 orders of mag
            chi2 = chi2 * 10 ** (np.round(12 - np.log10(chi2)))
        else:
            self.previous_cost_value = chi2

        return chi2

    def minimize_chi2(self):
        self.minimize_cost_function(cost_func='chi2')

# methods for maximum likelihood
    def log_likelihood(self, dx, dy, phi, fractions_sims, sigma=0):
        """
        Calculates the log likelihood for the given conditions.
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
        # generate sim pattern
        gen = self.pattern_generator
        # gen = PatternCreator(self.lib, self.XXmesh, self.YYmesh, simulations, mask=data_pattern.mask)
        sim_pattern = gen.make_pattern(dx, dy, phi, fractions_sims, total_events, sigma=sigma, type='ideal')
        self.sim_pattern = sim_pattern.copy()
        # negative log likelihood
        nll = -np.sum(data_pattern * np.log(sim_pattern))        # extended log likelihood - no need to fit events
        #ll = -np.sum(events_per_sim) + np.sum(data_pattern * np.log(sim_pattern))
        #print('likelihood - ', nll)
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
            self.verbose_graphics_ax.set_aspect('equal')
            self.verbose_graphics_fg.canvas.draw()
        # =====
        return nll

    def log_likelihood_call(self, params, enable_scale=False):
        # order of params is dx,dy,phi,total_cts,f_p1,f_p2,f_p3
        #print('params ', params)
        p0_scale = self._get_p0_scale() if enable_scale else np.ones(len(params))
        # print('p0_scale ', p0_scale)
        params_temp = ()
        di = 0
        for key in self._parameters_order:
            if self._parameters_dict[key]['use']:
                params_temp += (params[di] * p0_scale[di],)
                di += 1
            else:
                params_temp += (self._parameters_dict[key]['p0'],)
        #print('params_temp - ',params_temp)

        dx, dy, phi, total_cts, sigma = params_temp[0:5]

        fractions_sims = ()
        for i in range(self._n_sites):
            fractions_sims += (params_temp[5 + i],) #fractions f_p1, f_p2, f_p3,...
        #print('fractions_sims - ', fractions_sims, self._n_sites)

        nll = self.log_likelihood(dx, dy, phi, fractions_sims, sigma=sigma)

        # This is sort of a hack to avoid that after a very high chi2
        # the next point is not exactly the same as the previous.
        # This happens as if after a step in the direction of -gradient
        # The function value increases an intermediate step is chosen
        # at a position weigthed by the values at each extreme.
        # This allows the pattern to adjust better to the edges of the
        # experimental vs theoretical range.
        if self.previous_cost_value is None:
            self.previous_cost_value = nll
        elif nll > self.previous_cost_value * 1e12:
            # limit the increase to 12 orders of mag
            nll = nll * 10 ** (np.round(12 - np.log10(nll)))
        else:
            self.previous_cost_value = nll

        return nll

    def maximize_likelyhood(self):
        self.minimize_cost_function(cost_func='ml')

    def minimize_cost_function(self, cost_func='chi2'):

        if not cost_func in ('ml', 'chi2'):
            raise ValueError('cost function should be \'chi2\' or \'ml\'')

        if cost_func == 'ml':
            # total counts is not used in maximum likelyhood
            self._parameters_dict['total_cts']['use'] = False

        # set duplicated sites to zero
        self._fix_duplicated_sites()

        # order of params is dx,dy,phi,sigma,f_p1,f_p2,f_p3
        p0 = self._get_p0()
        #print('p0 - ', p0)

        # Parameter bounds
        bnds = self._get_bounds()

        # get patterns
        sites = ()
        for key in self._pattern_keys:
            if self._parameters_dict[key]['use']:
                sites += (self._parameters_dict[key]['value'],)
        #print('sites - ', sites)

        # generate sim pattern
        self.pattern_generator = PatternCreator(self._lib, self.XXmesh, self.YYmesh, sites,
                                                mask=self.data_pattern.mask,
                                                sub_pixels=self._parameters_dict['sub_pixels']['value'],
                                                mask_out_of_range = False)

        # defining cost function and get options
        if cost_func == 'chi2':
            function = self.chi_square_call
        elif cost_func == 'ml':
            function = self.log_likelihood_call

        # select method
        res = op.minimize(function, p0, args=True, method=self._minimization_method, bounds=bnds, \
                          options=self._fit_options)  # 'eps': 0.0001, L-BFGS-B
        if self._fit_options['disp']:
            print(res)
        # minimization with cobyla also seems to be a good option with {'rhobeg':1e-1/1e-2} . but it is unconstrained
        di = 0
        orientation_jac = np.zeros(3)
        for key in self._parameters_order:
            if self._parameters_dict[key]['use']:
                res['x'][di] *= self._parameters_dict[key]['scale']
                self._parameters_dict[key]['value'] = res['x'][di]
                res['jac'][di] /= self._parameters_dict[key]['scale']

                # setting up orientation_jac
                if key == 'dx':
                    orientation_jac[0] = res['jac'][di]
                elif key == 'dy':
                    orientation_jac[1] = res['jac'][di]
                elif key == 'phi':
                    orientation_jac[2] = res['jac'][di]

                di += 1
            else:
                self._parameters_dict[key]['value'] = self._parameters_dict[key]['p0']
        res['orientation jac'] = orientation_jac
        self.results = res

    def log_likelihood_call_explicit(self, dx, dy, phi, sigma, **kwargs):
        fractions_sims = ()
        for i in np.arange(1, 1 + self._n_sites):
            fraction = 'f_p' + str(i)
            fractions_sims += (kwargs.pop(fraction),)
        if kwargs:
            raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))

        # print('fractions_sims - ', fractions_sims)
        value = self.log_likelihood(dx, dy, phi, fractions_sims, sigma=sigma)
        # print('function value, ', value)
        return value

    def chi_square_call_explicit(self, dx, dy, phi, total_cts, sigma, **kwargs):
        fractions_sims = ()
        for i in np.arange(1, 1 + self._n_sites):
            fraction = 'f_p' + str(i)
            fractions_sims += (kwargs.pop(fraction),)
        if kwargs:
            raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))
        # print('fractions_sims - ', fractions_sims)
        value = self.chi_square(dx, dy, phi, total_cts, fractions_sims, sigma=sigma)
        # print('function value, ', value)
        return value


# methods for calculating error
    def get_std_from_hessian(self, x, enable_scale=True, func=''):
        x = np.array(x)
        #print('x', x)
        x /= self._get_p0_scale() if enable_scale else np.ones(len(x))
        #print('scaled x', x)
        if func == 'ml':
            f = lambda xx: self.log_likelihood_call(xx, enable_scale)
        elif func == 'chi2':
            f = lambda xx: self.chi_square_call(xx, enable_scale)
        else:
            raise ValueError('undefined function, should be likelihood or chi_square')
        H = nd.Hessian(f, step=1e-4)
        hh = H(x)
        #print('Parameters order', self._parameters_order)
        #print('Hessian diagonal', np.diag(hh))
        if np.linalg.det(hh) != 0:
            if func == 'ml':
                hh_inv = np.linalg.inv(hh)
            elif func == 'chi2':
                hh_inv = np.linalg.inv(0.5*hh)
            else:
                raise ValueError('undefined function, should be likelihood or chi_square')
            #print('np.diag(hh_inv)', np.diag(hh_inv))
            std = np.sqrt(np.diag(hh_inv))
            std *= self._get_p0_scale() if enable_scale else np.ones(len(x))
        else:
            warnings.warn('As Hessian is not invertible, errors are not calculated. '
                          'This usualy happens when all site fractions are zero')
            std = -np.ones(len(x))
        #print('errors,', std)
        self.std = std
        di = 0
        for key in self._parameters_order:
            if self._parameters_dict[key]['use']:
                self._parameters_dict[key]['std'] = std[di]
                di += 1
        return std

    def get_location_errors(self, params, simulations, func='', first=None, last=None, delta=None):
        warnings.warn('This function is broken, please ask eric to fix it if you need it.')
        return
        dx = params[0]
        dy = params[1]
        phi = params[2]
        events_rand = (params[3],)  # random
        events_per_sim = ()
        events_per_sim += (params[4],) if self._parameters_dict['pattern_1']['use'] else ()  # pattern 1
        events_per_sim += (params[5],) if self._parameters_dict['pattern_2']['use'] else ()  # pattern 2
        events_per_sim += (params[6],) if self._parameters_dict['pattern_3']['use'] else ()  # pattern 3
        # get patterns
        sims = ()
        sims += (simulations[0],) if self._parameters_dict['pattern_1']['use'] else ()
        sims += (simulations[1],) if self._parameters_dict['pattern_2']['use'] else ()
        sims += (simulations[2],) if self._parameters_dict['pattern_3']['use'] else ()
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



