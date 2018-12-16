

from pyfdd import Lib2dl
from pyfdd.lib2dl import read2dl

import matplotlib.pyplot as plt


def test_lib2dl(path):
    lib = Lib2dl(path)

    print_init_values(lib)

    lib.print_header()

    print(lib.get_simulations_list())

    plt.figure()
    imgmat = lib.get_simulation_patt(1)
    plt.contourf(imgmat)


def print_init_values(lib):
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
    path = "/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/FDD_libraries/GaN_89Sr/ue488g34.2dl"  # 89Sr [0001]
    # path = "/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/FDD_libraries/GaN_89Sr/ue567g54.2dl" #89Sr [-1102]
    # path = /home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/FDD_libraries/GaN_89Sr/ue646g53.2dl" #89Sr [-1101]

    test_lib2dl(path)

    plt.show()
