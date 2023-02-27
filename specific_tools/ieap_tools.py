import numpy as np
import matplotlib.pyplot as plt

def make_array_from_scatterdata(filename):
    data = np.loadtxt(filename)
    patternarray_h, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=512, range=((0, 512), (0, 512)))
    return patternarray_h.T


if __name__ == '__main__':
    filename = '/home/eric/data/IEAP-CTU/2022 work/ISOLDE Channeling/Data TPX3-QUAD/2022_10_09 121Sn in Diamond/data009/ScatterData-18-21 11-10-2022'
    data = np.loadtxt(filename)

    patternarray = make_array_from_scatterdata(filename)

    plt.imshow(patternarray, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.show()