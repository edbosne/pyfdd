#!/usr/bin/env python3

'''
Fit manager is the kernel class for fitting.
'''

__author__ = 'E. David-Bosne'
__email__ = 'eric.bosne@cern.ch'

from lib2dl import lib2dl
from patterncreator import *
from MedipixMatrix import *

class fitman:
    # pattern, library, pattern numbers, fit options
    def __init__(self):
        self.min_value = 10**12
        self.results = None
        self.best_fit = None
        pass

    def add_pattern(self,data_pattern,library):
        # should work with both the filename or object
        pass

    def run_fits(self,*args):
        # each input is a range of patterns to fit
        pass

    def save_output(self,filename):
        pass

    def get_pd_table(self):
        pass
