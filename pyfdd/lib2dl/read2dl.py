#!/usr/bin/env python3

"""
Source for the Lib2dl class. It allows to read .2dl files into a python dictionary or .json file.

The __main__ routine plots the first patterns
"""


__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

import struct
import numpy as np
import matplotlib.pyplot as plt
import io, json
import os
#import cPickle as json

import sys

import sys
if sys.version_info[0] < 3:
    raise "Must be using Python 3"

class read2dl:
    def __init__(self,filename):
        self.fileName = filename
        self.dict_2dl = {}
        self.short_sz = 2
        self.float_sz = 4

    def get_record(self, index, fileContent):
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

    def read_file(self):
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
        header, index = self.get_record(index, fileContent) #record size byte doesnt count for the size of the record
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

        #add here if it needs mirrorwing

        #end of header

        record_size = fileContent[index]
        #print(record_size, index)

        dict_spec = {}
        self.dict_2dl["Spectrums"] = ()
        while fileContent[index] != 130:
            record, index = self.get_record(index, fileContent)
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

            record, index = self.get_record(index, fileContent)

            nfloats = self.dict_2dl["nx"] * self.dict_2dl["ny"]
            #dict_spec["array"] = np.array(struct.unpack("<"+"f" * nfloats, record[0:nfloats * self.float_sz])).reshape((self.dict_2dl["ny"],self.dict_2dl["nx"])).tolist()
            self.dict_2dl["Spectrums"] += (dict_spec.copy(),)

    def get_array(self, spectrums_index):
        with open(self.fileName, mode='rb') as file: # b is important -> binary
            fileContent = file.read()
        index = spectrums_index #self.dict_2dl["Spectrums"][spectrums_index]["array_index"]
        record, index = self.get_record(index, fileContent)
        nfloats = self.dict_2dl["nx"] * self.dict_2dl["ny"]
        array = np.array(struct.unpack("<"+"f" * nfloats, record[0:nfloats * self.float_sz])).reshape((self.dict_2dl["ny"],self.dict_2dl["nx"]))
        return array.copy()

    def get_dict(self):
        return self.dict_2dl.copy()

    def save2json(self):
        # Save JSON File
        jsonfile = os.path.splitext(self.fileName)[0] + ".json"

        #print("writing to ", jsonfile)
        #print(os.path.splitext(self.fileName))
        # json save:
        # import json
        with io.open(jsonfile, 'w', encoding='utf-8') as f:
            f.write(str(json.dumps(self.dict_2dl, ensure_ascii=False, sort_keys=True, indent=4)),'utf-8')
        #with open(jsonfile, 'wb') as fp:
        #    json.dump(self.dict_2dl, fp) #can be used with pickle too

        # json load:
        #with open('data.json', 'r') as fp:
        #    data = json.load(fp)

        #print dict_spec["array"]
        #print self.dict_2dl

        #plt.contourf(dict_spec["array"])
        #plt.show()

    def list_simulations(self):
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


