

from pyfdd import lib2dl
from pyfdd.lib2dl import read2dl

import matplotlib.pyplot as plt


def test_lib2dl(path):
    lib = lib2dl(path)
    plt.figure()
    imgmat = lib.get_simulation_patt(1)
    plt.contourf(imgmat)

def test_read2dl(path):

    EClib = read2dl(path)
    EClib.read_file()
    ECdict = EClib.get_dict()

    EClib.print_header()
    print(EClib.list_simulations())

    nx = ECdict["nx"]
    ny = ECdict["ny"]
    xstep = ECdict["xstep"]
    ystep = ECdict["ystep"]

    plt.figure()
    imgmat = EClib.get_array(ECdict["Spectrums"][0]["array_index"]).reshape((ny, nx))
    plt.contourf(imgmat)


if __name__ == '__main__':
    path = "/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/FDD_libraries/GaN_89Sr/ue488g34.2dl"  # 89Sr [0001]
    # path = "/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/FDD_libraries/GaN_89Sr/ue567g54.2dl" #89Sr [-1102]
    # path = /home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/FDD_libraries/GaN_89Sr/ue646g53.2dl" #89Sr [-1101]

    test_read2dl(path)

    test_lib2dl(path)

    plt.show()
