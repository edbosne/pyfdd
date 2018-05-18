#!/usr/bin/env python3

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'


from .lib2dl import lib2dl

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
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
    def __init__(self, lib, xmesh=None, ymesh=None, simulations=0, mask=ma.nomask, sub_pixels=1, mask_out_of_range = True):
        """
        __init__ method for PatternCreator. Simulation and mesh are to be stated here.
        mask_out_of_range false means that points that are out of the range of simulations are not masked,
        :type lib: lib2dl
        :param lib: A lib2dl type object that points to the desired library
        :param xmesh: The horizontal mesh of the detector
        :param ymesh: The vertical mesh of the detector
        :param simulations: List of simulations to be used for creating the pattern. Order is kept as stated.
        """
        simulations = np.array(simulations)
        if len(simulations.shape) == 0: # ensures that dimention is 1
            simulations = simulations[np.newaxis]
        assert isinstance(lib, lib2dl)

        self.mask_out_of_range = mask_out_of_range

        # get original mesh if undefines
        if xmesh is None or ymesh is None:
            self._detector_xmesh = lib.XXmesh.copy()
            self._detector_ymesh = lib.YYmesh.copy()
        else:
            self._detector_xmesh = np.array(xmesh)
            self._detector_ymesh = np.array(ymesh)

        # expanded mesh
        assert isinstance(sub_pixels, int)
        self.sub_pixels = sub_pixels
        self._detector_xmesh_expanded = np.array([])
        self._detector_ymesh_expanded = np.array([])
        if self.sub_pixels > 1:
            self._expande_detector_mesh()

        # set mesh values
        self._nx_original = lib.nx
        self._ny_original = lib.ny
        self._xstep_original = lib.xstep
        self._ystep_original = lib.ystep
        self._xfirst_original = lib.xfirst
        self._yfirst_original = lib.yfirst
        self._xlast_original = lib.xlast
        self._ylast_original = lib.ylast

        # set working values
        self._nx = self._nx_original
        self._ny = self._ny_original
        self._xstep = self._xstep_original
        self._ystep = self._ystep_original
        self._xfirst = self._xfirst_original
        self._yfirst = self._yfirst_original
        self._xlast = self._xlast_original
        self._ylast = self._ylast_original

        # set working mesh for simulated pattern
        self._xmesh = np.array([])
        self._ymesh = np.array([])
        self._update_coordinates_mesh()
        self.sim_shape = self._xmesh.shape
        self.mask = mask

        # set orriginal pattern stack
        # TODO no stack
        # saving each pattern in a stack was used for individual normalization
        # this is not needed after all as all patterns should be normalized to random
        # or equivalently by the number of pixels in the pattern
        temp = np.ones(self.sim_shape)
        pattern_stack = temp[np.newaxis]
        for i in np.arange(simulations.size):
            temp = lib.get_simulation_patt(simulations[i])
            if not pattern_stack.size:
                pattern_stack = temp[np.newaxis].copy()
            else:
                pattern_stack = np.concatenate((pattern_stack, temp[np.newaxis]), 0)
        self._pattern_stack_original = pattern_stack.copy()
        self._pattern_current = np.ones(self._xmesh.shape)

        self.fractions_per_sim = np.zeros(simulations.size + 1) # +1 for random

    def make_pattern(self, dx, dy, phi, fractions_per_sim, total_events, sigma=0, type='ideal'):
        """
        Makes a pattern acoording to library and spectruns selected in the iniciation of the patterncreator
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
        assert fractions_per_sim.size == self._pattern_stack_original.shape[0], \
            'size of fractions_per_sim does not match the number of simulations'
        self._pattern_current = np.ones(self._xmesh.shape)
        # reset mesh
        self._xstep = self._xstep_original
        self._ystep = self._ystep_original
        self._xfirst = self._xfirst_original
        self._yfirst = self._yfirst_original
        self._xlast = self._xlast_original
        self._ylast = self._ylast_original
        self._update_coordinates_mesh()
        # apply fractions
        self._apply_fractions(fractions_per_sim)
        # gaussian convolution
        self._gaussian_conv(sigma)
        # rotate
        self._rotate(phi)
        # move mesh
        self._move(dx,dy)
        # render normalized pattern
        self._grid_interpolation(total_events)
        mask = self._pattern_current.mask.copy()
        sim_pattern = self._pattern_current
        # types
        if type == 'ideal':
            return sim_pattern
        elif type == 'montecarlo':
            n_total = total_events
            return ma.array(self._gen_mc_pattern(sim_pattern, n_total), mask=mask)
        elif type == 'poisson':
            return ma.array(np.random.poisson(sim_pattern), mask=mask)
        else:
            raise ValueError("invalid value for type: options are ideal, montecarlo and poisson")

    def _gen_mc_pattern(self,sim_pattern,n_total):
        """
        Generates a monte carlo pattern from a simulation pattern with n_total events
        The function normalizes the sim_pattern and uses it as a the PDF.
        :param sim_pattern: simulated pattern
        :param n_total: total number of events
        :return: montecarlo pattern
        """
        n_total = int(n_total)
        sim_pattern /= sim_pattern.sum()

        cdf = sim_pattern.reshape(-1).cumsum()
        inv_cdf = lambda value: np.searchsorted(cdf, value, side="left")
        mc_event = [inv_cdf(x) for x in np.random.uniform(0, 1, n_total)]
        mc_event_x = self._detector_xmesh.reshape(-1)[mc_event]
        mc_event_y = self._detector_ymesh.reshape(-1)[mc_event]
        bins = self._detector_xmesh.shape[::-1]
        range = [[self._detector_ymesh.min(), self._detector_ymesh.max()],
                 [self._detector_xmesh.min(), self._detector_xmesh.max()]]
        H, xedges, yedges = np.histogram2d(mc_event_y, mc_event_x, bins, range)
        return H

    def _apply_fractions(self, fractions):
        if not self._pattern_stack_original.shape[0] == fractions.size:
            raise ValueError('number of fractions is not the same as the number of simulations + rand')
        new_pattern = np.zeros(self.sim_shape)
        norm_factor = self.sim_shape[0] * self.sim_shape[1]
        for i in range(0, self._pattern_stack_original.shape[0]):
            new_pattern += (self._pattern_stack_original[i, :, :] / norm_factor) * fractions[i] # normalize to random
        self._pattern_current = new_pattern.copy()

    def _gaussian_conv(self, sigma=0):
        if sigma == 0:
            return
        assert self._xstep_original == self._ystep_original, 'Simulations steps are not the same in x and y'
        sigma_pix = sigma / self._xstep_original
        self._pattern_current = gaussian_filter(self._pattern_current, sigma_pix)

    def _rotate(self, ang=0):
        # Rotation
        """
        Rotates self._pattern_stack_original by ang
        :param ang: angle in degrees
        """
        # positive counterclockwise
        ang = -ang
        cval = 0 if self.mask_out_of_range else 1e-12
        self._pattern_current = rotate(self._pattern_current, ang, reshape=True, order=2, mode='constant',
                                       cval=cval, prefilter=False) #the order needs to be 2 for smoothness during fit

        # rotation can increase matrix size and therefore the mesh needs to be updates
        self._xfirst = self._xfirst_original - self._xstep_original * \
                                               0.5 * (self._pattern_current.shape[1] - self.sim_shape[1])
        self._yfirst = self._yfirst_original - self._ystep_original * \
                                               0.5 * (self._pattern_current.shape[0] - self.sim_shape[0])
        self._xlast = self._xlast_original + self._xstep_original * \
                                             0.5 * (self._pattern_current.shape[1] - self.sim_shape[1])
        self._ylast = self._ylast_original + self._ystep_original * \
                                             0.5 * (self._pattern_current.shape[0] - self.sim_shape[0])
        self._update_coordinates_mesh()

    def _update_coordinates_mesh(self):
        '''
        create or update object x and y mesh
        use for inicial mesh creation
        use to update mesh after a rotation with size change
        '''
        #set the stop between last and last+1step
        x = np.arange(self._xfirst, self._xlast + 0.5*self._xstep, self._xstep)
        y = np.arange(self._yfirst, self._ylast + 0.5*self._ystep, self._ystep)
        self._xmesh, self._ymesh = np.meshgrid(x, y)

    def _expande_detector_mesh(self):

        xstep = self._detector_xmesh[0, 1] - self._detector_xmesh[0, 0]
        ystep = self._detector_ymesh[1, 0] - self._detector_ymesh[0, 0]
        new_xstep = xstep / self.sub_pixels
        new_ystep = ystep / self.sub_pixels

        new_xfirst = self._detector_xmesh[0, 0] - xstep / 2 + new_xstep / 2
        new_yfirst = self._detector_ymesh[0, 0] - ystep / 2 + new_ystep / 2
        # there is no need to subtract newstep/2 as it should be added again in np.arange
        new_xlast = self._detector_xmesh[-1, -1] + xstep / 2 #- new_xstep / 2
        new_ylast = self._detector_ymesh[-1, -1] + ystep / 2 #- new_ystep / 2
        # just as in _update_coordinates_mesh
        x = np.arange(new_xfirst, new_xlast, new_xstep)
        y = np.arange(new_yfirst, new_ylast, new_ystep)
        self._detector_xmesh_expanded, self._detector_ymesh_expanded = np.meshgrid(x, y)

    def _move(self, dx=0, dy=0):
        '''
        Translation of pattern
        :param dx: translation in x, units in angle
        :param dy: translation in y, units in angle
        '''
        self._xmesh = self._xmesh + dx
        self._ymesh = self._ymesh + dy

    def _grid_interpolation(self, total_events, ):
        '''
        uses interpolation to get the values of the pattern at the grid positons
        it also normalizes each pattern to the previously set numbet or events for the given range
        instead they are substituted by a very small number 1e-12
        :return the updated pattern in the detector mesh
        '''

        '''
        fg = plt.figure(2)
        ax = fg.add_subplot(111)
        plt.ion()
        cont = None
        plt.contourf(self.mask)
        fg.canvas.draw()
        '''

        # convert to index space
        xscale = self._xmesh.shape[1] / (self._xmesh[0, -1] - self._xmesh[0, 0])
        yscale = self._ymesh.shape[0] / (self._ymesh[-1, 0] - self._ymesh[0, 0])
        if self.sub_pixels == 1:
            grid_x_temp = (self._detector_xmesh - self._xmesh[0, 0]) * xscale
            grid_y_temp = (self._detector_ymesh - self._ymesh[0, 0]) * yscale
        elif self.sub_pixels > 1:
            grid_x_temp = (self._detector_xmesh_expanded - self._xmesh[0, 0]) * xscale
            grid_y_temp = (self._detector_ymesh_expanded - self._ymesh[0, 0]) * yscale

        #interpolation
        temp_pattern = np.array([])
        cval = 0 if self.mask_out_of_range else 1e-12
        temp_pattern = map_coordinates(self._pattern_current, (grid_y_temp, grid_x_temp),
                                       order=1, prefilter=False, mode='constant', cval=cval)
        if self.sub_pixels > 1:
            y_final_size, x_final_size = self._detector_ymesh.shape
            factor = self.sub_pixels
            # sum over extra pixels, normalization here is redundant
            temp_pattern = temp_pattern.reshape([y_final_size, factor, x_final_size, factor]).sum(3).sum(1)
        temp_pattern = ma.array(data=temp_pattern, mask=self.mask)
        temp_pattern = temp_pattern / temp_pattern.sum() * total_events # number of events
        if self.mask_out_of_range:
            self._pattern_current = ma.masked_equal(temp_pattern, 0)
        else:
            self._pattern_current = ma.array(temp_pattern, mask=False)


if __name__ == "__main__":
    lib = lib2dl("/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_89Sr/ue567g54.2dl")
    #xmesh, ymesh = create_detector_mesh(22, 22, 1.4, 300)
    #xmesh, ymesh = create_detector_mesh(40, 40, 0.5, 300)
    xmesh, ymesh = create_detector_mesh(100, 100, 0.2, 300)
    # 5 subpixels is a good number for the pads
    gen = PatternCreator(lib)#, xmesh, ymesh, 1, sub_pixels=5)

    fractions_per_sim = np.array([0, 1])
    #fractions_per_sim /= fractions_per_sim.sum()
    total_events = 1
    pattern = gen.make_pattern(0.0, -0.0, -0, fractions_per_sim, total_events, sigma=0.1, type='ideal')
    print(pattern.sum())

    plt.figure(1)
    plt.contourf(xmesh, ymesh, pattern)
    plt.colorbar()

    plt.show()
