

import numpy as np
import matplotlib.pyplot as plt

def load_tpxquad_mask(filename, do_plot=False):
    assert isinstance(filename, str)

    pixelman_mask = np.loadtxt(filename)

    x = np.concatenate((np.arange(0, 256), np.arange(256, 512) + 2 * 2))
    X, Y = np.meshgrid(x, x)

    # chip0
    A = pixelman_mask[0:256, 0:256]
    AX = X[0:256, 0:256]
    AY = Y[0:256, 0:256]
    # chip1
    B = pixelman_mask[256:512, 0:256]
    BX = X[0:256, 256:512]
    BY = Y[0:256, 256:512]
    # chip2
    C = pixelman_mask[512:768, 0:256]
    CX = X[256:512, 256:512]
    CY = Y[256:512, 256:512]
    # chip3
    D = pixelman_mask[768:1024, 0:256]
    DX = X[256:512, 0:256]
    DY = Y[256:512, 0:256]

    Ar = A
    Br = B
    Cr = np.rot90(np.rot90(C))
    Dr = np.rot90(np.rot90(D))

    if do_plot:
        img = plt.figure()
        plt.pcolormesh(AX, AY, Ar, cmap=plt.cm.gray)
        plt.pcolormesh(BX, BY, Br, cmap=plt.cm.gray)
        plt.pcolormesh(CX, CY, Cr, cmap=plt.cm.gray)
        plt.pcolormesh(DX, DY, Dr, cmap=plt.cm.gray)
        ax = plt.axis('image')
        plt.show()

    temptop = np.concatenate((Ar, Br), 1)
    tempbot = np.concatenate((Dr, Cr), 1)
    mask_arr = np.concatenate((temptop, tempbot), 0)
    mask_arr = (mask_arr == 0).astype(np.int)
    return mask_arr

if __name__ == '__main__':
    filename = '/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/2015_GaN_24Na/2018 Analysis/TPX/jupyter/mask.txt'
    mask = load_tpxquad_mask(filename, do_plot=True)
    print(mask)