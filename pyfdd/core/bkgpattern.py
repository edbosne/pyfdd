

# Imports from standard library
import warnings

# Imports from 3rd party
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import gaussian_filter

import pyfdd
# Imports from project
from pyfdd.core.datapattern import DataPattern

class BkgPattern(DataPattern):
    def __init__(self, *, file_path=None, pattern_array=None, verbose=1,  **kwargs):
        super(BkgPattern, self).__init__(file_path=file_path, pattern_array=pattern_array, verbose=verbose,  **kwargs)

        self.smooth_sigma = 0
        self.correction_factor = 1
        # ===== Note on the correction factor =====
        # The factor is applied to the fractions resulting from the fit as:
        # Final fraction = Fitted fraction * Correction factor
        # From a gamma measurement the correction factor is estimated as
        # fractor = events in data / (events in data - expected gama events)

    def set_sigma(self, sigma):
        if sigma< 0:
            self.smooth_sigma = 0
        else:
            self.smooth_sigma = sigma

    def set_factor(self, new_factor):
        if not isinstance(new_factor, float):
            raise TypeError(f'Argument new_factor should be of type float, not {type(new_factor)}')

        self.correction_factor = new_factor

    def calculate_factor(self, bkg_time, data_time, data_cts):

        bkg_count_rate = self.pattern_matrix.sum() / bkg_time
        data_count_rate = data_cts / data_time

        self.correction_factor = data_count_rate / (data_count_rate - bkg_count_rate)

        return self.correction_factor

    def get_smoothed_background(self, as_datapattern=False):  #, measurement_dp=None, as_datapattern=False):

        # Check if pattern have been calibrated and have a simular range
        # if measurement_dp is None:
        #     measurement_dp=self
        # else:
        #     if not isinstance(DataPattern, measurement_dp):
        #         raise ValueError(f'Argument measurement_dp should be of type pyfdd.DataPattern, '
        #                          f'not {type(measurement_dp)}')
        #     # self.verify_ranges(measurement_dp)
        #     self.verify_shape(measurement_dp)

        # Smooth background
        smoothed_background = self._gaussian_conv()

        # Normalize to number of pixels
        #smoothed_background *= (smoothed_background.size / smoothed_background.sum())
        #assert smoothed_background.sum() == smoothed_background.size   # if this creates issues check the mask

        if as_datapattern is False:
            return smoothed_background
        else:
            dp = DataPattern(pattern_array=smoothed_background)
            dp.set_xymesh(self.xmesh, self.ymesh)
            return dp

    def verify_shape(self, measurement_dp:DataPattern):
        if not np.alltrue(measurement_dp.pattern_matrix.shape == self.pattern_matrix.shape):
            raise ValueError('The background pattern and the data pattern need '
                             'to have the same number of pixels and the same shape.'
                             'Please, ensure that the same manipulations are done in both patterns.')
        else:
            return True


    def verify_ranges(self, measurement_dp:DataPattern):
        if measurement_dp.is_mesh_defined is False or self.is_mesh_defined is False:
            raise ValueError('The angular mesh needs to be calibrated before background generation.')

        measurement_range = np.array([measurement_dp.xmesh.min(), measurement_dp.xmesh.max(),
                                      measurement_dp.ymesh.min(), measurement_dp.ymesh.max()])
        background_range = np.array([self.xmesh.min(), self.xmesh.max(),
                                     self.ymesh.min(), self.ymesh.max()])
        difference = np.abs(measurement_range - background_range)
        if difference.max() > 5:
            raise ValueError('There seems to be a large difference between the angular calibration of the'
                             'data and background patterns. Please verify the calibrations.')

        return True

    def _gaussian_conv(self):

        if self.is_mesh_defined is False:
            raise ValueError('The angular mesh needs to be calibrated before smoothing. '
                             'This is required as the sigma is in angular units.')

        if self.smooth_sigma < 0:
            self.smooth_sigma = 0
        if self.smooth_sigma == 0:
            return self.pattern_matrix
        x_step = self.xmesh[0, 1] - self.xmesh[0, 0]
        y_step = self.ymesh[1, 0] - self.ymesh[0, 0]
        xy_step = (x_step + y_step) / 2
        if not x_step == y_step:
            warnings.warn('Background pattern steps are not the same in x and y.\n'
                          'Gaussian convolution done assuming a step of {}'.format(xy_step))

        sigma_pix = self.smooth_sigma / xy_step
        smoothed_background = gaussian_filter(self.pattern_matrix, sigma_pix, truncate=3)
        return smoothed_background

