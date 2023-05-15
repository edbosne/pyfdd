

# Imports from standard library
import warnings

# Imports from 3rd party
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import gaussian_filter

import pyfdd
# Imports from project
from pyfdd.core.datapattern import DataPattern
from pyfdd.core.patterncreator import PatternCreator

class BackgroundTools:
    def __init__(self):

        self.smooth_sigma = 0

    def set_sigma(self, sigma):
        sigma = float(sigma)
        if not isinstance(sigma, float):
            raise TypeError(f'Argument sigma should be of type float, not {type(sigma)}')
        if not sigma >= 0:
            raise ValueError('Argument sigma needs to be larger or iqual to 0.')

        if sigma < 0:
            self.smooth_sigma = 0
        else:
            self.smooth_sigma = sigma

    @staticmethod
    def calculate_factor(bkg_cts, bkg_time, data_time, data_cts):

        # ===== Note on the correction factor =====
        # The factor is applied to the fractions resulting from the fit as:
        # Final fraction = Fitted fraction * Correction factor
        # From a gamma measurement the correction factor is estimated as
        # fractor = events in data / (events in data - expected gama events)
        if not(isinstance(bkg_cts, (float, int)) and bkg_cts >= 0 and \
               isinstance(bkg_time, (float, int)) and bkg_time >= 0 and \
               isinstance(data_time, (float, int)) and data_time > 0 and \
               isinstance(data_cts, (float, int)) and data_cts > 0):
            raise ValueError('Input values need to be positive.')

        if bkg_time == 0:
            return 1

        bkg_count_rate = bkg_cts / bkg_time
        data_count_rate = data_cts / data_time
        if bkg_count_rate >= data_count_rate:
            raise ValueError('The background count rate needs to be smaler than the data count rate.')

        corr_factor = data_count_rate / (data_count_rate - bkg_count_rate)
        corr_factor = np.round(corr_factor, decimals=2)
        return corr_factor


    def get_smoothed_background(self, background_pattern, as_datapattern=False):  #, measurement_dp=None, as_datapattern=False):
        if not isinstance(background_pattern, DataPattern):
            raise TypeError(f'Argument background_pattern should be of type pyfdd.DataPattern, '
                            f'not {type(background_pattern)}')

        if background_pattern.is_mesh_defined is False:
            raise ValueError('The angular mesh needs to be calibrated before smoothing. '
                             'This is required as the sigma is in angular units.')

        xmesh = background_pattern.xmesh
        ymesh = background_pattern.ymesh
        pattern_matrix = background_pattern.pattern_matrix

        # Smooth background
        smoothed_background = self._gaussian_conv(pattern_matrix, xmesh, ymesh)

        # Normalize to number of pixels
        #smoothed_background *= (smoothed_background.size / smoothed_background.sum())
        #assert smoothed_background.sum() == smoothed_background.size   # if this creates issues check the mask

        if as_datapattern is False:
            return smoothed_background
        else:
            dp = DataPattern(pattern_array=smoothed_background)
            dp.set_xymesh(xmesh, ymesh)
            return dp

    def _gaussian_conv(self, pattern_matrix, xmesh, ymesh):

        if self.smooth_sigma < 0:
            self.smooth_sigma = 0
        if self.smooth_sigma == 0:
            return pattern_matrix

        x_step = xmesh[0, 1] - xmesh[0, 0]
        y_step = ymesh[1, 0] - ymesh[0, 0]
        xy_step = (x_step + y_step) / 2
        if not x_step == y_step:
            warnings.warn('Background pattern steps are not the same in x and y.\n'
                          'Gaussian convolution done assuming a step of {}'.format(xy_step))

        sigma_pix = self.smooth_sigma / xy_step
        smoothed_background = gaussian_filter(pattern_matrix.astype(float), sigma_pix, truncate=3)
        return smoothed_background

    @staticmethod
    def verify_patterncreator(background_dp: DataPattern, creator: PatternCreator):
        if not np.alltrue(background_dp.pattern_matrix.shape == creator._detector_xmesh.shape):
            raise ValueError('The background pattern and pattern creator mesh need '
                             'to have the same number of pixels and the same shape.'
                             'Please, ensure that the same manipulations are done in both arrays.')
        else:
            return True

    @staticmethod
    def verify_mesh(background_array: np.ndarray, xmesh: np.ndarray, ymesh: np.ndarray):
        if (not np.alltrue(background_array.shape == xmesh.shape)) or \
                (not np.alltrue(background_array.shape == ymesh.shape)):
            raise ValueError('The background pattern and the x and y mesh need '
                             'to have the same number of pixels and the same shape.'
                             'Please, ensure that the same manipulations are done in both patterns.')
        else:
            return True

    @staticmethod
    def verify_shape(background_dp:DataPattern, measurement_dp:DataPattern):
        if not np.alltrue(measurement_dp.pattern_matrix.shape == background_dp.pattern_matrix.shape):
            raise ValueError('The background pattern and the data pattern need '
                             'to have the same number of pixels and the same shape.'
                             'Please, ensure that the same manipulations are done in both patterns.')
        else:
            return True

    @staticmethod
    def verify_ranges(background_dp:DataPattern, measurement_dp:DataPattern):
        if measurement_dp.is_mesh_defined is False or background_dp.is_mesh_defined is False:
            raise ValueError('The angular mesh needs to be calibrated before background generation.')

        measurement_range = np.array([measurement_dp.xmesh.min(), measurement_dp.xmesh.max(),
                                      measurement_dp.ymesh.min(), measurement_dp.ymesh.max()])
        background_range = np.array([background_dp.xmesh.min(), background_dp.xmesh.max(),
                                     background_dp.ymesh.min(), background_dp.ymesh.max()])
        difference = np.abs(measurement_range - background_range)
        if difference.max() > 5:
            raise ValueError('There seems to be a large difference between the angular calibration of the'
                             'data and background patterns. Please verify the calibrations.')

        return True


