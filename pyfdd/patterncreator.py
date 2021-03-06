#!/usr/bin/env python3

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'


from .lib2dl import Lib2dl

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import warnings
from scipy.ndimage.interpolation import rotate, map_coordinates
from scipy.interpolate import griddata, interpn
from scipy.ndimage import gaussian_filter


def create_detector_mesh(n_h_pixels, n_v_pixels, pixel_size, distance):
    """
    create a mesh for the detector.
    returns a xmesh matrix with the angular values of the detector in the horizontal axis
    and a ymesh matrix with the angular values of the detector in the vertical axis
    all distances must have the same units
    :param n_h_pixels: number of horizontal pixels
    :param n_v_pixels: number of vertical pixels
    :param pixel_size: size of pixel
    :param distance: distance from detector to sample
    :return: x and y mesh matrixes
    """
    d_theta = np.arctan(pixel_size/distance) * 180 / np.pi
    x_i = 0.5 * (n_h_pixels-1) * d_theta
    y_i = 0.5 * (n_v_pixels-1) * d_theta
    x = np.arange(n_h_pixels) * d_theta - x_i
    y = np.arange(n_v_pixels) * d_theta - y_i
    xmesh, ymesh = np.meshgrid(x,y)
    return xmesh, ymesh


class PatternCreator:
    '''
    Objects of this classe hold a single set of patterns and are used to create combined spectrums
    Can create patterns for a specific detector configuration
    Can create ideal patterns, patterns with poisson noise and by Monte Carlo
    '''
    def __init__(self, lib, xmesh=None, ymesh=None, simulations=None, mask=ma.nomask, sub_pixels=1,
                 mask_out_of_range=True):
        """
        __init__ method for PatternCreator. Simulation and mesh are to be stated here.
        mask_out_of_range false means that points that are out of the range of simulations are not masked,
        :type lib: Lib2dl
        :param lib: A lib2dl type object that points to the desired library
        :param xmesh: The horizontal mesh of the detector
        :param ymesh: The vertical mesh of the detector
        :param simulations: List of simulations to be used for creating the pattern. Order is kept as stated.
        """

        if simulations:
            simulations = np.array(simulations)
            if len(simulations.shape) == 0: # ensures that dimension is 1
                simulations = simulations[np.newaxis]
            self._n_sites = simulations.size
        else:
            self._n_sites = 0

        if not isinstance(lib, Lib2dl):
            raise ValueError('lib must be an instance of Lib2dl')
        self.lib = lib

        self._mask_out_of_range = mask_out_of_range

        # get original mesh if undefines
        if xmesh is None or ymesh is None:
            self._detector_xmesh = lib.XXmesh.copy()
            self._detector_ymesh = lib.YYmesh.copy()
        else:
            self._detector_xmesh = np.array(xmesh)
            self._detector_ymesh = np.array(ymesh)

        # expanded mesh
        assert isinstance(sub_pixels, int)
        self._sub_pixels = sub_pixels
        self._detector_xmesh_expanded = np.array([])
        self._detector_ymesh_expanded = np.array([])
        if self._sub_pixels > 1:
            self._expande_detector_mesh()

        # set simulation mesh values
        self._nx_lib2dl = lib.nx
        self._ny_lib2dl = lib.ny
        self._xstep_lib2dl = lib.xstep
        self._ystep_lib2dl = lib.ystep
        self._xfirst_lib2dl = lib.xfirst
        self._yfirst_lib2dl = lib.yfirst
        self._xlast_lib2dl = lib.xlast
        self._ylast_lib2dl = lib.ylast

        # set working mesh for simulated pattern
        self._xmesh = np.array([])
        self._ymesh = np.array([])
        self._update_coordinates_mesh()
        self._sim_shape = self._xmesh.shape
        self.mask = mask

        # set simulated patterns stack to avoid going back to the library
        # first pattern in the stack is the random
        temp = np.ones(self._sim_shape)
        pattern_stack = temp[np.newaxis]
        # loop to get each site pattern
        for i in np.arange(self._n_sites):
            temp = lib.get_simulation_patt(simulations[i])
            pattern_stack = np.concatenate((pattern_stack, temp[np.newaxis]), 0)
        self._pattern_stack = pattern_stack
        self._pre_smooth_pattern_stack = np.ones(self._pattern_stack.shape)
        self._pre_smooth_sigma = None
        self._pattern_current = np.ones(self._xmesh.shape)

        self.fractions_per_sim = np.zeros(self._n_sites + 1) # +1 for random

    def make_pattern(self, dx, dy, phi, fractions_per_site, total_events, sigma=0, type='ideal'):
        """
        Makes a pattern according to the library and sites selected in the initialization of the patterncreator
        Set total_events=1 and type='ideal' for a normalized spectrum.
        :param dx: delta x in angles
        :param dy: delta y in angles
        :param phi: delta phi in anlges
        :param fraction_per_sim: fractions of each simulated pattern, first is random
        :param total_events: total number of events
        :param type: 'ideal' for normalized pattern, 'montecarlo' for rand generated,
        'poisson' for ideal with poisson noise
        :return: masked array with pattern
        """
        fractions_per_site = np.array(fractions_per_site)
        if not fractions_per_site.size == self._n_sites:
            raise ValueError('Size of fractions_per_sim does not match the number of simulations')

        self._pattern_current = np.zeros(self._xmesh.shape)

        if self._sub_pixels > 1:
            self._detector_xmesh_temp = self._detector_xmesh_expanded.copy()
            self._detector_ymesh_temp = self._detector_ymesh_expanded.copy()
        else:
            self._detector_xmesh_temp = self._detector_xmesh.copy()
            self._detector_ymesh_temp = self._detector_ymesh.copy()

        #verify is pre-smothed patterns can be used to save time
        if self._pre_smooth_sigma is not None and self._pre_smooth_sigma == sigma:
            use_pre_smooth = True
        else:
            use_pre_smooth = False

        # don't change the order to the function calls
        # apply fractions
        random_fraction = np.array([1 - fractions_per_site.sum()])
        fractions = np.concatenate((random_fraction,fractions_per_site))
        self._apply_fractions(fractions, use_pre_smooth=use_pre_smooth)
        # gaussian convolution
        if not use_pre_smooth:
            self._gaussian_conv(sigma)
        # rotate
        self._rotate(phi)
        # move mesh
        self._move(dx, dy, phi)
        # render pattern
        self._grid_interpolation()
        if type == 'yield':
            return self._pattern_current.copy()

        # normalized pattern
        self._normalization(total_events)
        # keep mask for later
        mask = self._pattern_current.mask.copy()
        sim_pattern = self._pattern_current.copy()
        # types
        if type == 'ideal':
            return sim_pattern
        elif type == 'montecarlo':
            n_total = total_events
            return ma.array(self._gen_mc_pattern(sim_pattern, n_total), mask=mask)
        elif type == 'poisson':
            return ma.array(np.random.poisson(sim_pattern), mask=mask)
        else:
            raise ValueError("invalid value for type: options are ideal, yield, montecarlo and poisson")

    def pre_smooth_simulations(self, sigma):
        if sigma < 0:
            sigma = 0
        if not self._xstep_lib2dl == self._ystep_lib2dl:
            sim_step = (self._xstep_lib2dl + self._ystep_lib2dl) / 2
            warnings.warn('Simulations steps are not the same in x and y.\n'
                          'Gaussian convolution done assuming a step of {}'.format(sim_step))
        else:
            sim_step = self._xstep_lib2dl
        self._pre_smooth_sigma = sigma

        sigma_pix = sigma / sim_step
        for i in np.arange(1, self._n_sites+1): # index 1 is just the random
             # Truncating at 4 or at 2 causes that some Fit are unstable. Chose 3 as intermediate value
            self._pre_smooth_pattern_stack[i,:,:] = gaussian_filter(self._pattern_stack[i,:,:], sigma_pix, truncate=3)

    def _gen_mc_pattern(self, sim_pattern, n_total):
        """
        Generates a monte carlo pattern from a simulation pattern with n_total events
        The function normalizes the sim_pattern and uses it as a the PDF.
        :param sim_pattern_data: simulated pattern
        :param n_total: total number of events
        :return: montecarlo pattern
        """
        n_total = int(n_total)
        if isinstance(sim_pattern, ma.MaskedArray):
            sim_pattern_data = sim_pattern.data
            n_total = int(n_total * sim_pattern_data.sum() / sim_pattern.sum())
        else:
            sim_pattern_data = sim_pattern
        sim_pattern_data /= sim_pattern_data.sum()

        cdf = sim_pattern_data.reshape(-1).cumsum()
        inv_cdf = lambda value: np.searchsorted(cdf, value, side="left")
        mc_event = [inv_cdf(x) for x in np.random.uniform(0, 1, n_total)]
        mc_event_x = self._detector_xmesh.reshape(-1)[mc_event]
        mc_event_y = self._detector_ymesh.reshape(-1)[mc_event]
        bins = self._detector_xmesh.shape[::-1]
        range = [[self._detector_ymesh.min(), self._detector_ymesh.max()],
                 [self._detector_xmesh.min(), self._detector_xmesh.max()]]
        H, xedges, yedges = np.histogram2d(mc_event_y, mc_event_x, bins, range)
        return H

    def _apply_fractions(self, fractions, use_pre_smooth=False):

        if use_pre_smooth:
            pattern_stack = self._pre_smooth_pattern_stack
        else:
            pattern_stack = self._pattern_stack

        if not pattern_stack.shape[0] == fractions.size:
            raise ValueError('number of fractions is not the same as the number of simulations + rand')
        new_pattern = np.zeros(self._sim_shape)
        # Each pattern in the stack (simulation yields) is multiplied by the respective fraction
        for i in range(0, pattern_stack.shape[0]):
            new_pattern += pattern_stack[i, :, :] * fractions[i]
        self._pattern_current = new_pattern

    def _gaussian_conv(self, sigma=0):
        if sigma < 0:
            sigma = 0
        if sigma == 0:
            return
        if not self._xstep_lib2dl == self._ystep_lib2dl:
            sim_step = (self._xstep_lib2dl + self._ystep_lib2dl) / 2
            warnings.warn('Simulations steps are not the same in x and y.\n'
                          'Gaussian convolution done assuming a step of {}'.format(sim_step) )
        else:
            sim_step = self._xstep_lib2dl
        sigma_pix = sigma / sim_step
        # Truncating at 4 or at 2 causes that some Fit are unstable. Chose 3 as intermediate value
        self._pattern_current = gaussian_filter(self._pattern_current, sigma_pix, truncate=3)

    def _rotate(self, ang=0):

        # positive counterclockwise
        ang = -ang
        theta = np.radians(ang)
        c, s = np.cos(theta), np.sin(theta)
        # rotation matrix
        rot_matrix = np.array([[c, -s], [s, c]])

        x = self._detector_xmesh_temp.reshape(-1)
        y = self._detector_ymesh_temp.reshape(-1)
        xy = np.vstack((x,y))
        rotated_xy = rot_matrix.dot(xy)

        self._detector_xmesh_temp = rotated_xy[0, :].reshape(self._detector_xmesh_temp.shape)
        self._detector_ymesh_temp = rotated_xy[1, :].reshape(self._detector_ymesh_temp.shape)


    def _update_coordinates_mesh(self):
        '''
        create or update object x and y mesh
        use for inicial mesh creation
        use to update mesh after a rotation with size change
        '''
        #set the stop between last and last+1step
        x = np.arange(self._xfirst_lib2dl, self._xlast_lib2dl + 0.5*self._xstep_lib2dl, self._xstep_lib2dl)
        y = np.arange(self._yfirst_lib2dl, self._ylast_lib2dl + 0.5*self._ystep_lib2dl, self._ystep_lib2dl)
        self._xmesh, self._ymesh = np.meshgrid(x, y)

    def _expande_detector_mesh(self):

        xstep = self._detector_xmesh[0, 1] - self._detector_xmesh[0, 0]
        ystep = self._detector_ymesh[1, 0] - self._detector_ymesh[0, 0]
        new_xstep = xstep / self._sub_pixels
        new_ystep = ystep / self._sub_pixels

        new_xfirst = self._detector_xmesh[0, 0] - xstep / 2 + new_xstep / 2
        new_yfirst = self._detector_ymesh[0, 0] - ystep / 2 + new_ystep / 2
        # there is no need to subtract newstep/2 as it should be added again in np.arange
        new_xlast = self._detector_xmesh[-1, -1] + xstep / 2 #- new_xstep / 2
        new_ylast = self._detector_ymesh[-1, -1] + ystep / 2 #- new_ystep / 2
        # just as in _update_coordinates_mesh
        x = np.arange(new_xfirst, new_xlast, new_xstep)
        y = np.arange(new_yfirst, new_ylast, new_ystep)
        self._detector_xmesh_expanded, self._detector_ymesh_expanded = np.meshgrid(x, y)

    def _move(self, dx=0, dy=0, ang=0):
        '''
        Translation of pattern
        :param dx: translation in x, units in angle
        :param dy: translation in y, units in angle
        :param ang: angle of previous rotation. (x,y) vector also needs to be rotated
        '''

        # positive counterclockwise
        ang = -ang
        theta = np.radians(ang)
        c, s = np.cos(theta), np.sin(theta)
        # rotation matrix
        rot_matrix = np.array([[c, -s], [s, c]])

        xy = np.vstack((dx, dy))
        dx, dy = rot_matrix.dot(xy)

        self._detector_xmesh_temp = self._detector_xmesh_temp - dx
        self._detector_ymesh_temp = self._detector_ymesh_temp - dy

    def _grid_interpolation(self):
        '''
        uses interpolation to get the values of the pattern at the grid positons
        it also normalizes each pattern to the previously set numbet or events for the given range
        instead they are substituted by a very small number 1e-12
        :return the updated pattern in the detector mesh
        '''

        # convert to index space
        xscale = self._xmesh.shape[1] / (self._xmesh[0, -1] - self._xmesh[0, 0])
        yscale = self._ymesh.shape[0] / (self._ymesh[-1, 0] - self._ymesh[0, 0])

        grid_x_temp = (self._detector_xmesh_temp - self._xmesh[0, 0]) * xscale
        grid_y_temp = (self._detector_ymesh_temp - self._ymesh[0, 0]) * yscale

        #interpolation
        cval = 0 if self._mask_out_of_range else 1e-12
        temp_pattern = map_coordinates(self._pattern_current, (grid_y_temp, grid_x_temp),
                                       order=2, prefilter=False, mode='constant', cval=cval)

        if self._sub_pixels > 1:
            y_final_size, x_final_size = self._detector_ymesh.shape
            factor = self._sub_pixels

            # correct for out of range
            out_of_range_correction = np.array(temp_pattern == cval, int)
            out_of_range_correction = out_of_range_correction. \
                reshape([y_final_size, factor, x_final_size, factor]).sum(3).sum(1)
            out_of_range_correction = out_of_range_correction > 0

            # sum over extra pixels, normalization here is redundant
            temp_pattern = temp_pattern.reshape([y_final_size, factor, x_final_size, factor]).sum(3).sum(1)
            temp_pattern /= factor ** 2
            temp_pattern[out_of_range_correction] = cval

        temp_pattern = ma.array(data=temp_pattern, mask=self.mask)

        if self._mask_out_of_range:
            temp_pattern = ma.masked_equal(temp_pattern, 0)

        self._pattern_current = temp_pattern

    def _normalization(self, total_events=1):

        temp_pattern = self._pattern_current.data / self._pattern_current.sum() * total_events  # number of events
        temp_pattern = ma.array(data=temp_pattern, mask=self._pattern_current.mask)
        self._pattern_current = temp_pattern


