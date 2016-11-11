#!/usr/bin/env python3



__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'


import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate, map_coordinates
from scipy.interpolate import griddata, interpn

from read2dl import read2dl


class lib2dl:
    '''
    The lib2dl object holds the 2dl library and is used to produce patterns from the manybeam simulations to be then fitted with experimental data.
    '''

    def __init__(self,filename):
        '''
        init method for lib2dl
        :param filename: string, name of file
        '''
        # TODO verify filename
        EClib = read2dl(filename)
        EClib.read_file()
        self.ECdict = EClib.get_dict()

        self.nx = self.ECdict["nx"]
        self.ny = self.ECdict["ny"]
        self.xstep = self.ECdict["xstep"]
        self.ystep = self.ECdict["ystep"]
        self.xfirst = self.ECdict["xfirst"]
        self.yfirst = self.ECdict["yfirst"]
        self.xlast = self.ECdict["xlast"]
        self.ylast = self.ECdict["ylast"]
        self.numdim = self.nx * self.ny
        self.xmirror = False
        self.ymirror = False
        self.check_mirror()
        self.XXmesh = np.array([])
        self.YYmesh = np.array([])
        self.create_coordinates_mesh()
        self.pattern_current = np.ones((self.ny_mirror, self.nx_mirror))
        self.pattern_stack = self.pattern_current[np.newaxis]
        self.mask = np.zeros((self.ny_mirror,self.nx_mirror))
        self.pattern_current = ma.array(self.pattern_current, mask=self.mask)
        self.keepasoriginals()

        self.sim_list = EClib.list_simulations()

    def keepasoriginals(self):
        '''
        store needed to reproduce starting pattern and mesh i.e. before any rotation or translation
        '''
        self.nx_original = self.nx
        self.ny_original = self.ny
        self.xstep_original = self.xstep
        self.ystep_original = self.ystep
        self.xfirst_original =  self.xfirst
        self.yfirst_original = self.yfirst
        self.xlast_original = self.xlast
        self.ylast_original = self.ylast
        self.numdim_original = self.numdim
        self.pattern_original = self.pattern_current.copy()
        self.mask_original = self.mask.copy()
        self.XXmesh_original = self.XXmesh.copy()
        self.YYmesh_original = self.YYmesh.copy()
        self.pattern_stack_original = self.pattern_stack.copy()

    def check_mirror(self):
        '''
        Decide if the spectra should be mirrored in x or y direction
        '''
        if self.xlast == 0:
            self.xmirror = True
            self.nx_mirror = self.nx * 2-1
            self.xlast = (self.nx_mirror-1) * self.xstep + self.xfirst
        else:
            self.xmirror = False
            self.nx_mirror = self.nx

        if self.ylast == 0:
            self.ymirror = True
            self.ny_mirror = self.ny * 2-1
            self.ylast = (self.ny_mirror-1) * self.ystep + self.yfirst
        else:
            self.ymirror = False
            self.ny_mirror = self.ny

    def create_coordinates_mesh(self, keepasoriginal=True):
        '''
        create or update object x and y mesh
        use for inicial mesh creation with keepasoriginal True
        use to update mesh after a rotation with size change with keepasoriginal false
        '''
        x = np.arange(self.xfirst, self.xlast + self.xstep, self.xstep)
        y = np.arange(self.yfirst, self.ylast + self.ystep, self.ystep)
        self.XXmesh, self.YYmesh = np.meshgrid(x,y)
        if keepasoriginal:
            self.XXmesh_original = self.XXmesh.copy()
            self.YYmesh_original = self.YYmesh.copy()

    def mirror(self,pattern):
        # expand if needs to me mirrored
        new_pattern = pattern.copy()
        if self.xmirror:
            new_pattern = np.concatenate((np.fliplr(new_pattern), new_pattern[:][1:]), 1)
        if self.ymirror:
            new_pattern = np.concatenate((np.flipud(new_pattern), new_pattern[1:][:]), 0)
        return new_pattern

    def get_patterns(self,rnd_frac=1, sim_1=None, frac_1=0, sim_2=None, frac_2=0, sim_3=None, frac_3=0):
        factor = self.ny * self.nx
        self.set_patterns_counts(rnd_frac * factor, sim_1, frac_1 * factor, sim_2, frac_2 * factor, sim_3, frac_3 * factor)

    def set_patterns_counts(self, cts_rnd=1, sim_1=None, cts_1=0, sim_2=None, cts_2=0, sim_3=None, cts_3=0):
        '''
        gets up to tree patterns plus the random and saves them in a stack.
        patterns are normalized to the input number of counts
        the supperposition of all patterns is saved in currect pattern
        '''
        self.cnts_stack = (cts_rnd, cts_1, cts_2, cts_3)
        temp = np.ones((self.ny, self.nx))
        temp *= cts_rnd / (self.ny * self.nx)
        temp = self.mirror(temp)
        self.pattern_stack = temp[np.newaxis]
        if sim_1 is not None:
            temp = np.array(self.ECdict["Spectrums"][sim_1]["array"]).copy()
            temp *= cts_1 / temp.sum()
            temp = self.mirror(temp)
            self.pattern_stack = np.concatenate((self.pattern_stack, temp[np.newaxis]), 0)
        if sim_2 is not None:
            temp = np.array(self.ECdict["Spectrums"][sim_2]["array"]).copy()
            temp *= cts_2 / temp.sum()
            temp = self.mirror(temp)
            self.pattern_stack = np.concatenate((self.pattern_stack, temp[np.newaxis]), 0)
        if sim_3 is not None:
            temp = np.array(self.ECdict["Spectrums"][sim_2]["array"]).copy()
            temp *= cts_3 / temp.sum()
            temp = self.mirror(temp)
            self.pattern_stack = np.concatenate((self.pattern_stack, temp[np.newaxis]), 0)
        print(self.pattern_stack.shape)
        self.pattern_stack_original = self.pattern_stack.copy()

    def rotate(self, ang=0):
        # Rotation
        # TODO raise exeption stack not properly set
        new_pattern_stact = np.array([])
        for i in range(self.pattern_stack_original.shape[0]):
            pattern_i = self.pattern_stack_original[i,:,:].copy()
            if not new_pattern_stact.size:
                new_pattern_stact = \
                    rotate(pattern_i, ang, reshape=True, order=3, mode='constant', cval=0, prefilter=True)[np.newaxis]
            else:
                new_pattern_stact = np.concatenate((new_pattern_stact, \
                    rotate(pattern_i, ang, reshape=True, order=3, mode='constant', cval=0, prefilter=True)[np.newaxis]), 0)
        self.pattern_stack = new_pattern_stact.copy()
        # rotation can increase matrix size and therefore the mesh needs to be updates
        self.xfirst = self.xfirst_original - self.xstep_original * \
                      0.5 * (self.pattern_stack.shape[2] - self.pattern_original.shape[1])
        self.yfirst = self.yfirst_original - self.ystep_original * \
                      0.5 * (self.pattern_stack.shape[1] - self.pattern_original.shape[0])
        self.xlast = self.xlast_original + self.xstep_original * \
                      0.5 * (self.pattern_stack.shape[2] - self.pattern_original.shape[1])
        self.ylast = self.ylast_original + self.ystep_original * \
                      0.5 * (self.pattern_stack.shape[1] - self.pattern_original.shape[0])
        # TODO check if mesh size and pattern size are the same
        self.create_coordinates_mesh(keepasoriginal=False)
        print('shapes of mesh', self.XXmesh.shape,self.YYmesh.shape)
        print('shape of pattern',new_pattern_stact.shape)
        print('ylast-yfirst/step', (self.ylast - self.yfirst)/self.ystep_original)

    def move(self, dx=0, dy=0):
        # Translation
        self.XXmesh = self.XXmesh + dx
        self.YYmesh = self.YYmesh + dy

    def get_current_pattern(self):
        self.pattern_current = self.pattern_stack.sum(0)
        self.pattern_current = ma.masked_equal(self.pattern_current, 0)
        if not self.pattern_current.mask.shape == self.pattern_current.shape:
            self.pattern_current = ma.masked_array(self.pattern_current, np.zeros(self.pattern_current.shape))
        self.mask = self.pattern_current.mask
        return self.pattern_current
        #print(self.pattern_current)
        #print(self.mask)

    def grid_interpolation(self,grid_x,grid_y,totalcounts=0):
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
        if False: # slow method
            mask = self.mask.reshape(-1)
            points = (self.XXmesh.reshape(-1)[~mask], self.YYmesh.reshape(-1)[~mask])
            values = self.pattern_current.reshape(-1)[~mask]
            #if not totalcounts == 0:
            #    values *= totalcounts / values.sum()
            return ma.masked_equal(griddata(points, values, (grid_x, grid_y), method='cubic'),0)
        if True: # fast method
            grid_x_temp = (grid_x - self.XXmesh[0, 0]) * self.XXmesh.shape[1] / (self.XXmesh[0,-1] - self.XXmesh[0,0])
            grid_y_temp = (grid_y - self.YYmesh[0, 0]) * self.YYmesh.shape[0] / (self.YYmesh[-1, 0] - self.YYmesh[0, 0])
            new_pattern_stact = np.array([])
            for i in range(self.pattern_stack.shape[0]):
                if not new_pattern_stact.size:
                    temp_pattern = map_coordinates(self.pattern_stack[i,:,:], (grid_y_temp, grid_x_temp),
                                                   order=1, prefilter=True, mode='constant', cval=0)[np.newaxis]
                    if not temp_pattern.sum() == 0:
                        new_pattern_stact = (temp_pattern/temp_pattern.sum())*self.cnts_stack[i]
                else:
                    temp_pattern = map_coordinates(self.pattern_stack[i, :, :], (grid_y_temp, grid_x_temp),
                                                   order=1, prefilter=True, mode='constant', cval=0)[np.newaxis]
                    if not temp_pattern.sum() == 0:
                        temp_pattern = (temp_pattern/temp_pattern.sum())*self.cnts_stack[i]
                        new_pattern_stact = np.concatenate((new_pattern_stact,temp_pattern),0)
            if new_pattern_stact.size == 0:
                new_pattern_stact = np.zeros(grid_y_temp.shape)[np.newaxis]
            return ma.masked_equal(new_pattern_stact.sum(0),0)
            #rtrn_data = map_coordinates(self.pattern_current, (grid_y_temp, grid_x_temp),order=3,prefilter=False)
            #return ma.masked_equal(rtrn_data,0)


if __name__ == "__main__":
    lib = lib2dl("/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_89Sr/ue567g54.2dl")
    #lib.get_patterns(0.5,0,0.5)
    lib.set_patterns_counts(100000, 0, 900000)

    lib.rotate(-5)
    lib.move(0.5, -0.5)
    #print(lib.pattern_current.view(ma.MaskedArray))

    plt.figure(0)
    plt.contourf(lib.XXmesh, lib.YYmesh, lib.get_current_pattern())
    plt.colorbar()

    x = np.arange(-3, 2.1, 0.1)
    y = np.arange(-3, 2.1, 0.1)
    XXmesh, YYmesh = np.meshgrid(x, y)

    plt.figure(1)
    plt.contourf(XXmesh, YYmesh, lib.grid_interpolation(XXmesh,YYmesh))
    plt.colorbar()

    plt.show(block=True)
