#!/usr/bin/env python3

"""
Source for the read2dl class. It allows to read .2dl files into a python dictionary or .json file.

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
        self.fileContent = str('')
        self.dict_2dl = {}

    def get_record(self, index):
        #assume being at beggining of record
        record = bytearray()
        record_size = self.fileContent[index]
        index += 1
        if record_size == 130:
            print("end of file")
            return
        while record_size == 129:
            record += self.fileContent[index:index+128] #last byte and first byte of record are of no use
            index = index + record_size
            record_size = self.fileContent[index]
            #print "one pass", record_size, index
            index += 1
        record += self.fileContent[index:index+record_size]
        next_index = index + record_size + 1
        return record, next_index

    def read_file(self):
        with open(self.fileName, mode='rb') as file: # b is important -> binary
            self.fileContent = file.read()

        index = 0

        if self.fileContent[0] == 75:
            print("2dl file opened")
        else:
            print("this is not a 2dl file")
            return

        index = 1
        record_size = self.fileContent[index]
        index += 1
        #print(record_size, index)

        index = 1
        header, index = self.get_record(index) #record size byte doesnt count for the size of the record
        #print("after header index is - ", index)
        rec_index = 0
        short_sz = 2
        float_sz = 4
        self.dict_2dl["nx"] = struct.unpack("<h",header[rec_index:rec_index+short_sz])[0]
        rec_index += short_sz
        self.dict_2dl["ny"] = struct.unpack("<h",header[rec_index:rec_index+short_sz])[0]
        rec_index += short_sz
        rec_index += float_sz*2 #not used
        self.dict_2dl["xstep"] = round(struct.unpack("<f",header[rec_index:rec_index+float_sz])[0],6)
        rec_index += float_sz
        self.dict_2dl["ystep"] = round(struct.unpack("<f",header[rec_index:rec_index+float_sz])[0],6)
        rec_index += float_sz
        self.dict_2dl["xfirst"] = struct.unpack("<f",header[rec_index:rec_index+float_sz])[0]
        rec_index += float_sz
        self.dict_2dl["yfirst"] = struct.unpack("<f",header[rec_index:rec_index+float_sz])[0]
        rec_index += float_sz
        self.dict_2dl["xlast"] = ((self.dict_2dl["nx"]-1)*self.dict_2dl["xstep"])+self.dict_2dl["xfirst"]
        self.dict_2dl["ylast"] = ((self.dict_2dl["ny"]-1)*self.dict_2dl["ystep"])+self.dict_2dl["yfirst"]

        #add here if it needs mirrorwing

        #end of header

        record_size = self.fileContent[index]
        #print(record_size, index)

        dict_spec = {}
        self.dict_2dl["Spectrums"] = ()
        while self.fileContent[index] != 130:
            record, index = self.get_record(index)
            rec_index = 0
            dict_spec["Spectrum number"] = struct.unpack("<h",record[rec_index:rec_index+short_sz])[0]
            rec_index += short_sz
            dict_spec["Spectrum_description"] = (record[rec_index:rec_index + 50]).decode('utf-8')
            rec_index += 50
            dict_spec["factor"] = struct.unpack("<f",record[rec_index:rec_index+float_sz])[0]
            rec_index += float_sz
            dict_spec["u2"] = struct.unpack("<f",record[rec_index:rec_index+float_sz])[0]
            rec_index += float_sz
            dict_spec["sigma"] = struct.unpack("<f",record[rec_index:rec_index+float_sz])[0]
            rec_index += float_sz

            record_size = self.fileContent[index]
            #print record_size, index

            record, index = self.get_record(index)

            nfloats = self.dict_2dl["nx"] * self.dict_2dl["ny"]
            dict_spec["array"] = np.array(struct.unpack("<"+"f" * nfloats,record[0:nfloats * float_sz])).reshape((self.dict_2dl["ny"],self.dict_2dl["nx"])).tolist()
            self.dict_2dl["Spectrums"] += (dict_spec.copy(),)
        # empty filecontent
        self.fileContent = str('')

    def get_dict(self):
        return self.dict_2dl

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
                 spectrum["u2"], \
                 spectrum["sigma"]])
            ll = np.concatenate((ll, l[np.newaxis]), axis=0) if ll.size else l[np.newaxis]
        return ll.tolist()

def print_header(dict_2dl):
    print("nx, ny - ", dict_2dl["nx"], ", ", dict_2dl["ny"])
    print("xstep, ystep - ", dict_2dl["xstep"], ", ", dict_2dl["ystep"])
    print("xfirst, yfirst - ", dict_2dl["xfirst"], ", ", dict_2dl["yfirst"])
    print("xlast, ylast - ", dict_2dl["xlast"], ", ", dict_2dl["ylast"])


if __name__ == "__main__":
    EClib = read2dl("/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_89Sr/ue488g34.2dl") #89Sr [0001]
    #EClib = read2dl("/home/eric/cernbox/Channeling_analysis/FDD_libraries/GaN_89Sr/ue567g54.2dl") #89Sr [-1102]
    #EClib = read2dl("/home/eric/cernbox/FDD_libraries/ue646g53.2dl") #89Sr [-1101]

    EClib.read_file()
    ECdict = EClib.get_dict()
    #print_header(ECdict)
    #print(EClib.list_simulations())
    nx = ECdict["nx"]
    ny = ECdict["ny"]
    xstep = ECdict["xstep"]
    ystep = ECdict["ystep"]
    numdim = nx*ny
    print(nx, ny)
    print(numdim)

    y=[]
    for spec in ECdict["Spectrums"]:
        y += [np.sum(np.array(spec["array"]).reshape((ny, nx)))]
    print(y)
    plt.ylabel('Total sum')
    plt.xlabel('Pattern #')
    plt.plot(y)

    plt.show()


    figN = 0
    for i in range(0,2):
        plt.figure(figN)
        figN += 1
        imgmat = np.array(ECdict["Spectrums"][i]["array"]).reshape((ny, nx))[:,30::]
        plt.contourf(imgmat)
    #plt.show()
