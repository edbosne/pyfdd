

from pyfdd import Lib2dl, PatternCreator, DataPattern, FitManager
from pyfdd.patterncreator import create_detector_mesh

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    lib = Lib2dl("/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/FDD_libraries/GaN_89Sr/ue567g54.2dl")
    #xmesh, ymesh = create_detector_mesh(22, 22, 1.4, 300)
    #xmesh, ymesh = create_detector_mesh(40, 40, 0.5, 300)
    xmesh, ymesh = create_detector_mesh(100, 100, 0.3, 300)
    print('xmesh shape', xmesh.shape)
    # 5 subpixels is a good number for the pads
    gen = PatternCreator(lib, xmesh, ymesh, 1, sub_pixels=5, mask_out_of_range=False)

    fractions_per_sim = np.array([1])#[0, 1])
    #fractions_per_sim /= fractions_per_sim.sum()
    total_events = 1
    pattern = gen.make_pattern(0.0, -0.5, -5, fractions_per_sim, total_events, sigma=0.1, type='ideal')
    print(pattern.sum())

    plt.figure(1)
    plt.contourf(xmesh, ymesh, pattern)
    plt.colorbar()

    plt.show()
