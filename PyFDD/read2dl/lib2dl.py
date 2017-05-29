#!/usr/bin/env python3



__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'


import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate, map_coordinates
from scipy.interpolate import griddata, interpn

from .read2dl import read2dl


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
        self.EClib = read2dl(filename)
        self.EClib.read_file()
        self.ECdict = self.EClib.get_dict()

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
        #self.create_coordinates_mesh()
        self.pattern_current = np.ones((self.ny_mirror, self.nx_mirror))
        self.pattern_stack = self.pattern_current[np.newaxis]
        self.mask = np.zeros((self.ny_mirror,self.nx_mirror))
        self.pattern_current = ma.array(self.pattern_current, mask=self.mask)

        self.sim_list = self.EClib.list_simulations()


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

    def get_simulation_patt(self,num):
        assert num >= 1, 'pattern number must be positive'
        assert num <= len(self.ECdict["Spectrums"]), 'pattern number is not valid'
        temp = self.EClib.get_array(self.ECdict["Spectrums"][num-1]["array_index"])
        #temp = np.array(self.ECdict["Spectrums"][num]["array_index"]).copy()
        return self.mirror(temp)

    def mirror(self, pattern):
        # expand if needs to me mirrored
        new_pattern = pattern.copy()
        if self.xmirror:
            new_pattern = np.concatenate((np.fliplr(new_pattern), new_pattern[:,1:]), 1)
        if self.ymirror:
            new_pattern = np.concatenate((np.flipud(new_pattern), new_pattern[1:,:]), 0)
        return new_pattern


if __name__ == "__main__":
    lib = lib2dl("/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_89Sr/ue488g34.2dl")  # 89Sr [0001]
    plt.figure(1)
    imgmat = lib.get_simulation_patt(1)
    plt.contourf(imgmat)
    plt.show()