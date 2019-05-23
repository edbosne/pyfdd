

from pyfdd import Lib2dl, PatternCreator, DataPattern, FitManager
from pyfdd.patterncreator import create_detector_mesh

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    lib = Lib2dl("/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/FDD_libraries/GaN_89Sr/ue567g54.2dl")
    #xmesh, ymesh = create_detector_mesh(22, 22, 1.4, 300)
    #xmesh, ymesh = create_detector_mesh(40, 40, 0.5, 300)
    xmesh, ymesh = create_detector_mesh(200, 200, 0.15, 300)
    print('xmesh shape', xmesh.shape)
    gen = PatternCreator(lib, xmesh, ymesh, simulations=10, sub_pixels=5, mask_out_of_range=True)

    plt.figure()
    imgmat = lib.get_simulation_patt(10)
    plt.contourf(imgmat)
    plt.colorbar()

    fractions_per_sim = np.array([1])#[0, 1])
    #fractions_per_sim /= fractions_per_sim.sum()
    total_events = 1
    pattern = gen.make_pattern(0.0, -0.1, -6, fractions_per_sim, total_events, sigma=0.1, type='yield')
    print(pattern.sum())
    print(pattern[50,50])

    plt.figure()
    plt.contourf(xmesh, ymesh, pattern)
    plt.colorbar()

    plt.show()
