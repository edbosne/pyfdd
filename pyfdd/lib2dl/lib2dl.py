#!/usr/bin/env python3

"""
Source of the Lib2dl class. It allows to read .2dl files into a python dictionary or .json file.

"""


__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

import struct
import io, json
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate, map_coordinates
from scipy.interpolate import griddata, interpn

class Lib2dl:
    '''
    The Lib2dl object holds the 2dl library and is used to produce patterns from the manybeam simulations to be then fitted with experimental data.
    '''

    def __init__(self,filename):
        '''
        init method for Lib2dl
        :param filename: string, name of file
        '''

        self.fileName = filename    # Name of the .2dl library file
        self.dict_2dl = {}          # Dictionary to store the whole .2dl file, except the simulation arrays
        self.short_sz = 2           # Size of short
        self.float_sz = 4           # Size of float

        # read .2dl file
        self._read_file()

        # set handy values
        self.nx = self.dict_2dl["nx"]
        self.ny = self.dict_2dl["ny"]
        self.xstep = self.dict_2dl["xstep"]
        self.ystep = self.dict_2dl["ystep"]
        self.xfirst = self.dict_2dl["xfirst"]
        self.yfirst = self.dict_2dl["yfirst"]
        self.xlast = self.dict_2dl["xlast"]
        self.ylast = self.dict_2dl["ylast"]
        self.numdim = self.nx * self.ny
        self.xmirror = False
        self.ymirror = False
        self._check_mirror()
        xrange = np.arange(self.xfirst, self.xlast+self.xstep, self.xstep)
        yrange = np.arange(self.yfirst, self.ylast + self.ystep, self.ystep)
        self.XXmesh, self.YYmesh = np.meshgrid(xrange, yrange)

        # if it doest break remove
        #self.pattern_current = np.ones((self.ny_mirror, self.nx_mirror))
        #self.pattern_stack = self.pattern_current[np.newaxis]
        #self.mask = np.zeros((self.ny_mirror,self.nx_mirror))
        #self.pattern_current = ma.array(self.pattern_current, mask=self.mask)

    def _get_record(self, index, fileContent):
        '''
        Gets record from the fortran binary file
        :param index: Position of the record
        :param fileContent: content of the .2dl file
        :return:
        '''
        #assume being at beggining of record
        record = bytearray()
        record_size = fileContent[index]
        index += 1
        if record_size == 130:
            print("end of file")
            return
        while record_size == 129:
            record += fileContent[index:index+128] #last byte and first byte of record are of no use
            index = index + record_size
            record_size = fileContent[index]
            #print "one pass", record_size, index
            index += 1
        record += fileContent[index:index+record_size]
        next_index = index + record_size + 1
        return record, next_index

    def _read_file(self):
        '''
        read the .2dl file
        :return:
        '''
        with open(self.fileName, mode='rb') as file: # b is important -> binary
            fileContent = file.read()

        index = 0

        if fileContent[0] == 75:
            pass
            #print("2dl file opened")
        else:
            print("this is not a 2dl file")
            return

        index = 1
        record_size = fileContent[index]
        index += 1
        #print(record_size, index)

        index = 1
        header, index = self._get_record(index, fileContent) #record size byte doesnt count for the size of the record
        #print("after header index is - ", index)
        rec_index = 0
        self.dict_2dl["nx"] = struct.unpack("<h",header[rec_index:rec_index+self.short_sz])[0]
        rec_index += self.short_sz
        self.dict_2dl["ny"] = struct.unpack("<h",header[rec_index:rec_index+self.short_sz])[0]
        rec_index += self.short_sz
        rec_index += self.float_sz*2 #not used
        self.dict_2dl["xstep"] = round(struct.unpack("<f",header[rec_index:rec_index+self.float_sz])[0],6)
        rec_index += self.float_sz
        self.dict_2dl["ystep"] = round(struct.unpack("<f",header[rec_index:rec_index+self.float_sz])[0],6)
        rec_index += self.float_sz
        self.dict_2dl["xfirst"] = struct.unpack("<f",header[rec_index:rec_index+self.float_sz])[0]
        rec_index += self.float_sz
        self.dict_2dl["yfirst"] = struct.unpack("<f",header[rec_index:rec_index+self.float_sz])[0]
        rec_index += self.float_sz
        self.dict_2dl["xlast"] = ((self.dict_2dl["nx"]-1)*self.dict_2dl["xstep"])+self.dict_2dl["xfirst"]
        self.dict_2dl["ylast"] = ((self.dict_2dl["ny"]-1)*self.dict_2dl["ystep"])+self.dict_2dl["yfirst"]

        #end of header

        record_size = fileContent[index]
        #print(record_size, index)

        dict_spec = {}
        self.dict_2dl["Spectrums"] = ()
        while fileContent[index] != 130:
            record, index = self._get_record(index, fileContent)
            rec_index = 0
            dict_spec["Spectrum number"] = struct.unpack("<h",record[rec_index:rec_index+self.short_sz])[0]
            rec_index += self.short_sz
            dict_spec["Spectrum_description"] = (record[rec_index:rec_index + 50]).decode('utf-8')
            rec_index += 50
            # rounding for numerical accuracy
            dict_spec["factor"] = round(struct.unpack("<f",record[rec_index:rec_index+self.float_sz])[0],6)
            rec_index += self.float_sz
            dict_spec["u1"] = round(struct.unpack("<f",record[rec_index:rec_index+self.float_sz])[0],6)
            rec_index += self.float_sz
            dict_spec["sigma"] = round(struct.unpack("<f",record[rec_index:rec_index+self.float_sz])[0],6)
            rec_index += self.float_sz

            record_size = fileContent[index]
            #print record_size, index

            dict_spec["array_index"] = index

            record, index = self._get_record(index, fileContent)

            nfloats = self.dict_2dl["nx"] * self.dict_2dl["ny"]
            self.dict_2dl["Spectrums"] += (dict_spec.copy(),)

    def _check_mirror(self):
        '''
        Decide if the spectra should be mirrored in x or y direction
        '''
        if self.xlast == 0:
            self.xmirror = True
            self.nx_mirror = self.nx * 2 - 1
            self.xlast = (self.nx_mirror - 1) * self.xstep + self.xfirst
        else:
            self.xmirror = False
            self.nx_mirror = self.nx

        if self.ylast == 0:
            self.ymirror = True
            self.ny_mirror = self.ny * 2 - 1
            self.ylast = (self.ny_mirror - 1) * self.ystep + self.yfirst
        else:
            self.ymirror = False
            self.ny_mirror = self.ny

    def _mirror(self, pattern):
        '''
        expands the pattern if it needs to be mirrored
        :param pattern:
        :return:
        '''
        # expand if needs to me mirrored
        new_pattern = pattern.copy()
        if self.xmirror:
            new_pattern = np.concatenate((np.fliplr(new_pattern), new_pattern[:,1:]), 1)
        if self.ymirror:
            new_pattern = np.concatenate((np.flipud(new_pattern), new_pattern[1:,:]), 0)
        return new_pattern

    def get_dict(self):
        return self.dict_2dl.copy()

    def save2json(self):
        '''
        Saves the dictionary (library without the patterns) into json file
        :return:
        '''
        # Save JSON File
        jsonfile = os.path.splitext(self.fileName)[0] + ".json"

        with io.open(jsonfile, 'w', encoding='utf-8') as f:
            f.write(str(json.dumps(self.dict_2dl, ensure_ascii=False, sort_keys=True, indent=4)),'utf-8')

        #with open(jsonfile, 'wb') as fp:
        #    json.dump(self.dict_2dl, fp) #can be used with pickle too

        # json load:
        #with open('data.json', 'r') as fp:
        #    data = json.load(fp)

    def get_simulations_list(self):
        '''
        Returns a list containing all simulations description
        :return:
        '''
        ll = np.array([])
        for spectrum in self.dict_2dl["Spectrums"]:
            l = np.array([spectrum["Spectrum number"],\
                 spectrum["Spectrum_description"],\
                 spectrum["factor"], \
                 spectrum["u1"], \
                 spectrum["sigma"]])
            ll = np.concatenate((ll, l[np.newaxis]), axis=0) if ll.size else l[np.newaxis]
        return ll.tolist()

    def print_header(self):
        print("nx, ny - ", self.dict_2dl["nx"], ", ", self.dict_2dl["ny"])
        print("xstep, ystep - ", self.dict_2dl["xstep"], ", ", self.dict_2dl["ystep"])
        print("xfirst, yfirst - ", self.dict_2dl["xfirst"], ", ", self.dict_2dl["yfirst"])
        print("xlast, ylast - ", self.dict_2dl["xlast"], ", ", self.dict_2dl["ylast"])

    def get_simulation_patt(self, num):
        '''
        Get simulation with of number num
        :param num: index of the pattern +1
        :return:
        '''
        if not num >= 1:
            raise ValueError('pattern number must be positive')
        if not num <= len(self.dict_2dl["Spectrums"]):
            raise ValueError('pattern number is not valid')

        assert num == self.dict_2dl["Spectrums"][num - 1]["Spectrum number"]

        record = None

        # get array from file
        with open(self.fileName, mode='rb') as file: # b is important -> binary
            fileContent = file.read()
            record_index = self.dict_2dl["Spectrums"][num - 1]["array_index"]
            record, next_index = self._get_record(record_index, fileContent)

        # convert to floats and reshape
        nfloats = self.dict_2dl["nx"] * self.dict_2dl["ny"]
        array_long = np.array(struct.unpack("<"+"f" * nfloats, record[0:nfloats * self.float_sz]))
        array_temp = array_long.reshape((self.dict_2dl["ny"],self.dict_2dl["nx"]))

        # mirror if needed
        array_mirror = self._mirror(array_temp)

        return array_mirror


