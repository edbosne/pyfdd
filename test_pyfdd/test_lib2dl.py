

import os
import unittest
import numpy as np
import time

from pyfdd import Lib2dl




class TestLib2dl(unittest.TestCase):

    def test_lib2dl(self):
        start = time.time()
        lib = Lib2dl('data_files/sb600g05.2dl')
        total_time = time.time() - start
        print(f'loading the library took {total_time*1000}ms')

        self.print_init_values(lib)

        lib.print_header()

        print(lib.get_simulations_list())

    def print_init_values(self, lib):
        print('\n\n\nprinting init values \n')
        print('lib.fileName, lib.dict_2dl, lib.short_sz, lib.float_sz')
        print(lib.fileName, lib.dict_2dl, lib.short_sz, lib.float_sz)

        print('lib.nx, lib.ny, lib.xstep, lib.ystep')
        print(lib.nx, lib.ny, lib.xstep, lib.ystep)

        print('lib.xfirst, lib.yfirst, lib.xlast, lib.ylast')
        print(lib.xfirst, lib.yfirst, lib.xlast, lib.ylast)

        print('lib.numdim, lib.xmirror, lib.ymirror')
        print(lib.numdim, lib.xmirror, lib.ymirror)

        print('lib.XXmesh, lib.YYmesh')
        print(lib.XXmesh, lib.YYmesh)

        print('\n\n\n')


if __name__ == '__main__':
    unittest.main()
