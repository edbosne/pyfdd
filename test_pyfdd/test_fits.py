from pyfdd import lib2dl, Fit, DataPattern, PatternCreator
from pyfdd.fit import create_detector_mesh

import math
import numpy as np

analysis_path = "/home/eric/cernbox/University/CERN-projects/Betapix/Analysis/Channeling_analysis/"
lib_path = analysis_path + "FDD_libraries/GaN_24Na/ue567g29.2dl"
mm_path = analysis_path + "2015_GaN_24Na/2018 Analysis/TPX/RT/-1102/pattern_d3_Npix0-20_rebin2x2_180_100k.json"


def make_mc_pattern(lib, sites, size = 'medium'):

    if size == 'pad':
        # pad pixel size
        xmesh, ymesh = create_detector_mesh(20, 20, 1.4, 300)
    elif size == 'medium':
        # medium pixel size
        xmesh, ymesh = create_detector_mesh(50, 50, 0.5, 300)
    elif size == 'timepix':
        # timepix pixel size
        xmesh, ymesh = create_detector_mesh(400, 400, 0.055, 300)
    elif size == 'vsmall':
        # very small
        x = np.arange(-1.79, 1.8, 0.01)
        xmesh, ymesh = np.meshgrid(x, x)

    creator = PatternCreator(lib, xmesh, ymesh, (1,65))#(249-249+1,377-249+1))
    fractions_per_sim = np.array([0.60, 0.30, 0.10])  # first is random
    total_events = 1e6
    patt = creator.make_pattern(-0.08, 0.18, 5, fractions_per_sim, total_events, sigma=0.1, type='poisson')
    return patt, xmesh, ymesh


def test_chi2_fit(lib, xmesh, ymesh, patt):

    ft = Fit(lib)
    ft.verbose_graphics = True

    plt.figure(0)
    plt.contourf(xmesh, ymesh, patt)  # , np.arange(0, 3000, 100))
    plt.colorbar()
    plt.show(block=False)

    # set a fitting routine
    counts_ordofmag = 10 ** (int(math.log10(patt.sum())))
    ft.set_data_pattern(xmesh, ymesh, patt)
    # ft.set_patterns_to_fit(249-249,377-249)
    ft.set_patterns_to_fit(1, 65)  # ,129)
    ft.parameters_dict['sub_pixels']['value'] = 1

    ft.set_scale_values(dx=1, dy=1, phi=1, total_cts=counts_ordofmag, sigma=0.1, f_p1=1)
    ft.set_inicial_values(-1, 0.5, 180, counts_ordofmag, sigma=0.1)
    ft.set_optimization_profile('fine')
    # ft.set_inicial_values(mm.center[0], mm.center[1], mm.angle, counts_ordofmag, sigma=0.1)
    ft.minimize_chi2()
    print(ft.results)
    print('sigma in sim step units - ', ft.results['x'][4] / lib.xstep)
    print('Calculating errors ...')
    # var = ft.get_variance_from_hessian(ft.results['x'], enable_scale=False, func='chi_square')
    # print('var - ', var)
    # ft.print_variance(ft.res['x'],var)
    # x = res['x'] * ft.p0_scale[0:5]
    # ft.set_scale_values()
    # # There is a warning because the hessian starts with a step too big, don't worry about it
    # H = nd.Hessian(ft.log_likelihood_call)#,step=1e-9)
    # hh = H(x)
    # print(hh)
    # print(np.linalg.inv(hh))
    # ft.set_scale_values(dx=1, dy=1, phi=10, f_rand=counts_ordofmag, f_p1=counts_ordofmag)
    # ft.print_results(res,hh)

    print('data points ', np.sum(~patt.mask))

    return ft


def test_ml_fit(lib, xmesh, ymesh, patt):

    ft = Fit(lib)
    ft.verbose_graphics = True

    # set a fitting routine
    counts_ordofmag = 10 ** (int(math.log10(patt.sum())))
    ft.set_data_pattern(xmesh, ymesh, patt)
    # ft.set_patterns_to_fit(249-249,377-249)
    ft.set_patterns_to_fit(1, 65)  # ,129)
    ft.parameters_dict['sub_pixels']['value'] = 1

    ft.set_scale_values(dx=1, dy=1, phi=1, total_cts=-1, f_p1=1, f_p2=1)
    ft.set_inicial_values(-1, 0.5, 0, -1, sigma=0.1)
    ft.fix_parameters(False, False, False, False, False, False, False, False)
    ft.set_optimization_profile('fine')
    # ft.set_inicial_values(mm.center[0], mm.center[1], mm.angle, -1, sigma=0.1)
    ft.maximize_likelyhood()
    print(ft.results)
    print('sigma in sim step units - ', ft.results['x'][4] / lib.xstep)
    print('Calculating errors ...')
    # var = ft.get_variance_from_hessian(ft.results['x'], enable_scale=False, func='likelihood')
    # print('var - ', var)
    # ft.print_variance(ft.res['x'],var)
    # ft.get_location_errors(res['x'], (0,), last=300, func='likelihood')$

    print('data points ', np.sum(~patt.mask))

    return ft


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_chi2_min = False
    test_likelihood_max = False

    lib = lib2dl(lib_path)

    # set a pattern to fit
    # mc
    # patt, xmesh, ymesh = make_mc_pattern(lib)

    mm = DataPattern(file_path=mm_path)
    patt = mm.matrixOriginal
    xmesh = mm.xmesh
    ymesh = mm.ymesh

    plt.figure(0)
    plt.contourf(xmesh, ymesh, patt)  # , np.arange(0, 3000, 100))
    plt.colorbar()
    plt.show(block=False)

    if test_chi2_min:
        ft = test_chi2_fit(lib, xmesh, ymesh, patt)

    if test_likelihood_max:
        ft = test_ml_fit(lib, xmesh, ymesh, patt)

    plt.figure(2)
    plt.contourf(xmesh, ymesh, ft.sim_pattern)
    plt.colorbar()

    plt.figure(3)
    if test_chi2_min:
        plt.contourf(xmesh, ymesh, ft.sim_pattern - patt)
    if test_likelihood_max:
        plt.contourf(xmesh, ymesh, ft.sim_pattern - patt / patt.sum())
    plt.colorbar()
    plt.show(block=True)
