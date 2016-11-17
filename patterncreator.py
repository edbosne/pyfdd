#!/usr/bin/env python3

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'


from lib2dl import lib2dl

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate, map_coordinates
from scipy.interpolate import griddata, interpn

def create_detector_mesh(n_h_pixels, n_v_pixels, pixel_size, distance):
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
    def __init__(self, lib, xmesh=None, ymesh=None, simulations=0):
        simulations = np.array(simulations)
        xmesh = np.array(xmesh)
        ymesh = np.array(ymesh)
        assert isinstance(lib, lib2dl)

        # get original mesh
        if xmesh is None or ymesh is None:
            self._detector_xmesh = lib.XXmesh.copy()
            self._detector_ymesh = lib.YYmesh.copy()
        else:
            self._detector_xmesh = xmesh.copy()
            self._detector_ymesh = ymesh.copy()

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

        # set working mesh
        self._xmesh = np.array([])
        self._ymesh = np.array([])
        self._update_coordinates_mesh()
        self.sim_shape = self._xmesh.shape

        # set orriginal pattern stack
        temp = np.ones(self._xmesh.shape)
        pattern_stack = temp[np.newaxis]
        for i in np.arange(simulations.size):
            temp = lib.get_simulation_patt(i)
            if not pattern_stack.size:
                pattern_stack = temp[np.newaxis].copy()
            else:
                pattern_stack = np.concatenate((pattern_stack, temp[np.newaxis]), 0)
        self._pattern_stack_original = pattern_stack
        self._pattern_stack = pattern_stack

        self.events_per_sim = np.zeros(simulations.size+1) # +1 for random

    def make_pattern(self, dx, dy, phi, events_per_sim, type='ideal'):
        self.events_per_sim = events_per_sim
        # reset mesh
        self._xstep = self._xstep_original
        self._ystep = self._ystep_original
        self._xfirst = self._xfirst_original
        self._yfirst = self._yfirst_original
        self._xlast = self._xlast_original
        self._ylast = self._ylast_original
        self._update_coordinates_mesh()
        # rotate
        self._rotate(phi)
        # move mesh
        self._move(dx,dy)
        # render normalized pattern
        sim_pattern = self._grid_interpolation()
        mask = sim_pattern.mask
        # types
        if type == 'ideal':
            return sim_pattern
        elif type == 'montecarlo':
            n_total = self.events_per_sim.sum()
            return ma.array(self._gen_mc_pattern(sim_pattern, n_total), mask=mask)
        elif type == 'poisson':
            return ma.array(np.random.poisson(sim_pattern), mask=mask)
        else:
            raise ValueError("invalid value for type: options are ideal, montecarlo and poisson")

    def _gen_mc_pattern(self,sim_pattern,n_total):
        n_total = int(n_total)
        sim_pattern /= sim_pattern.sum()
        cdf = sim_pattern.reshape(-1).cumsum()
        inv_cdf = lambda value: np.searchsorted(cdf, value, side="left")
        mc_event = [inv_cdf(x) for x in np.random.uniform(0, 1, n_total)]
        mc_event_x = self._detector_xmesh.reshape(-1)[mc_event]
        mc_event_y = self._detector_ymesh.reshape(-1)[mc_event]
        H, xedges, yedges = np.histogram2d(mc_event_y, mc_event_x, self._detector_xmesh.shape[::-1])
        return H

    def _rotate(self, ang=0):
        # Rotation
        new_pattern_stact = np.array([])
        for i in range(self._pattern_stack_original.shape[0]):
            pattern_i = self._pattern_stack_original[i, :, :].copy()
            temp = rotate(pattern_i, ang, reshape=True, order=3, mode='constant', cval=0, prefilter=True)
            if not new_pattern_stact.size:
                new_pattern_stact = temp[np.newaxis].copy()
            else:
                new_pattern_stact = np.concatenate((new_pattern_stact, temp[np.newaxis]), 0)
        self._pattern_stack = ma.masked_equal(new_pattern_stact,0)

        # rotation can increase matrix size and therefore the mesh needs to be updates
        self._xfirst = self._xfirst_original - self._xstep_original * \
                                               0.5 * (self._pattern_stack.shape[2] - self.sim_shape[1])
        self._yfirst = self._yfirst_original - self._ystep_original * \
                                               0.5 * (self._pattern_stack.shape[1] - self.sim_shape[0])
        self._xlast = self._xlast_original + self._xstep_original * \
                                             0.5 * (self._pattern_stack.shape[2] - self.sim_shape[1])
        self._ylast = self._ylast_original + self._ystep_original * \
                                             0.5 * (self._pattern_stack.shape[1] - self.sim_shape[0])
        self._update_coordinates_mesh()

    def _update_coordinates_mesh(self):
        '''
        create or update object x and y mesh
        use for inicial mesh creation with keepasoriginal True
        use to update mesh after a rotation with size change with keepasoriginal false
        '''
        #set the stop between last and last+1step
        x = np.arange(self._xfirst, self._xlast + 0.5*self._xstep, self._xstep)
        y = np.arange(self._yfirst, self._ylast + 0.5*self._ystep, self._ystep)
        self._xmesh, self._ymesh = np.meshgrid(x, y)

    def _move(self, dx=0, dy=0):
        # Translation
        self._xmesh = self._xmesh + dx
        self._ymesh = self._ymesh + dy

    def _grid_interpolation(self):
        '''
        uses interpolation to get the values of the pattern at the grid positons
        it also normalizes each pattern to the previously set numbet or events for the given range
        '''
        '''
        fg = plt.figure(2)
        ax = fg.add_subplot(111)
        plt.ion()
        cont = None
        plt.contourf(self.mask)
        fg.canvas.draw()
        '''
        # # slow method
        # mask = self.mask.reshape(-1)
        # points = (self.XXmesh.reshape(-1)[~mask], self.YYmesh.reshape(-1)[~mask])
        # values = self.pattern_current.reshape(-1)[~mask]
        # #if not totalcounts == 0:
        # #    values *= totalcounts / values.sum()
        # return ma.masked_equal(griddata(points, values, (grid_x, grid_y), method='cubic'),0)

        # fast method
        # convert to index space
        xscale = self._xmesh.shape[1] / (self._xmesh[0, -1] - self._xmesh[0, 0])
        yscale = self._ymesh.shape[0] / (self._ymesh[-1, 0] - self._ymesh[0, 0])
        grid_x_temp = (self._detector_xmesh - self._xmesh[0, 0]) * xscale
        grid_y_temp = (self._detector_ymesh - self._ymesh[0, 0]) * yscale
        new_pattern_stact = np.array([])
        temp_pattern = np.array([])
        for i in range(self._pattern_stack.shape[0]):
            temp_pattern = map_coordinates(self._pattern_stack[i, :, :], (grid_y_temp, grid_x_temp),
                                           order=1, prefilter=True, mode='constant', cval=0)[np.newaxis]
            sum = temp_pattern.sum()
            if not sum == 0:
                temp_pattern = (temp_pattern / sum) * self.events_per_sim[i]
            else:
                temp_pattern = np.zeros(temp_pattern.shape)[np.newaxis]
            if not new_pattern_stact.size:
                new_pattern_stact = temp_pattern.copy()
            else:
                new_pattern_stact = np.concatenate((new_pattern_stact,temp_pattern),0)
        return ma.masked_equal(new_pattern_stact.sum(0),0)
        #rtrn_data = map_coordinates(self.pattern_current, (grid_y_temp, grid_x_temp),order=3,prefilter=False)
        #return ma.masked_equal(rtrn_data,0)


if __name__ == "__main__":
    lib = lib2dl("/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_89Sr/ue567g54.2dl")
    xmesh, ymesh = create_detector_mesh(22, 22, 1.4, 300)
    xmesh, ymesh = create_detector_mesh(40, 40, 0.5, 300)
    gen = PatternCreator(lib, xmesh, ymesh, 0)
    events_per_sim = np.array([1.5, 0.5]) * 1e6
    pattern = gen.make_pattern(0.5, -0.5, 0.25, events_per_sim, 'ideal')

    plt.figure(1)
    plt.contourf(xmesh, ymesh, pattern)
    plt.colorbar()

    plt.show()