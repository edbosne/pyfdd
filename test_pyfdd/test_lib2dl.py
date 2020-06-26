

from pyfdd import Lib2dl

import matplotlib.pyplot as plt
import time


def test_lib2dl(path):
    start = time.time()
    lib = Lib2dl(path)
    total_time = time.time() - start
    print(f'loading the library took {total_time*1000}ms')

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
    path = "/home/eric/Desktop/last files/Janni/Eric/nc111cf.2dl"  # 89Sr [0001]

    test_lib2dl(path)

    plt.show()
