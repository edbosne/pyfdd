import os
import numpy as np

import pyfdd
from pyfdd.core.datapattern import create_detector_mesh


def make_pad_db(sites, orientation, fractions, total_events, savename=None):

    # Define detector
    n_h_pixels = n_v_pixels = 22
    pixel_size = 1.3  # mm
    distance = 315  # mm
    xmesh, ymesh = create_detector_mesh(n_h_pixels, n_v_pixels, pixel_size, distance)

    # Define Pattern Creator
    pc = pyfdd.PatternCreator(lib, xmesh, ymesh, sites, sub_pixels=8, mask_out_of_range=True)

    # Get orientation
    dx = orientation['dx']
    dy = orientation['dy']
    phi = orientation['phi']

    #total_events = 1e6  # use 1 for p.d.f., ignored if pattern_type = 'yield'
    sigma = 0.05
    pattern_type = 'montecarlo'  # use ideal, yield, montecarlo and poisson

    # Create a data pattern
    dp = pc.make_datapattern(dx, dy, phi, fractions, total_events, sigma=sigma, pattern_type=pattern_type)

    # Save
    if savename is not None:
        dp.io_save_json(savename)
        print('saved', savename)

    return dp


def make_tpx_quad_db(sites, orientation, fractions, total_events, savename=None):
    # Define detector
    n_h_pixels = n_v_pixels = 516
    pixel_size = 0.055  # mm
    distance = 315  # mm
    xmesh, ymesh = create_detector_mesh(n_h_pixels, n_v_pixels, pixel_size, distance)

    # Define Pattern Creator
    pc = pyfdd.PatternCreator(lib, xmesh, ymesh, sites, sub_pixels=1, mask_out_of_range=True)

    # Get orientation
    dx = orientation['dx']
    dy = orientation['dy']
    phi = orientation['phi']

    # total_events = 1e6  # use 1 for p.d.f., ignored if pattern_type = 'yield'
    sigma = 0.05
    pattern_type = 'montecarlo'  # use ideal, yield, montecarlo and poisson

    # Create a data pattern
    dp = pc.make_datapattern(dx, dy, phi, fractions, total_events, sigma=sigma, pattern_type=pattern_type)

    # Save
    if savename is not None:
        dp.io_save_json(savename)
        print('saved', savename)

    return dp


def make_tpx_quad_txt(sites, orientation, fractions, total_events, savename=None):

    dp = make_tpx_quad_db(sites, orientation, fractions, total_events)

    data_matrix = dp.pattern_matrix.data.copy()

    data_matrix = np.delete(data_matrix, range(256, 260), 0)
    data_matrix = np.delete(data_matrix, range(256, 260), 1)

    # Save
    if savename is not None:
        np.savetxt(savename, data_matrix, "%d")
        print('saved', savename)


if __name__ == '__main__':

    # Import library
    print('curdir', os.getcwd())
    test_files_path = ""
    lib_path = os.path.join(test_files_path, "sb600g05.2dl")
    lib = pyfdd.Lib2dl(lib_path)

    # Define sites
    simulations_n = [1, 23]  # site 1 - S, site 23 - H
    fractions = [0.3, 0.2]

    # Define orientation
    orientation = {'dx': -0.1,
                   'dy': 0.2,
                   'phi': 3}

    # Make and save data patterns
    savename = os.path.join(test_files_path, "pad_dp_2M.json")
    make_pad_db(simulations_n, orientation, fractions, total_events=2e6, savename=savename)

    savename = os.path.join(test_files_path, "tpx_quad_dp_2M.json")
    make_tpx_quad_db(simulations_n, orientation, fractions, total_events=2e6, savename=savename)

    savename = os.path.join(test_files_path, "tpx_quad_dp_100K.json")
    make_tpx_quad_db(simulations_n, orientation, fractions, total_events=1e5, savename=savename)

    # Make and save txt patterns
    savename = os.path.join(test_files_path, "tpx_quad_array_2M.txt")
    make_tpx_quad_txt(simulations_n, orientation, fractions, total_events=2e6, savename=savename)

