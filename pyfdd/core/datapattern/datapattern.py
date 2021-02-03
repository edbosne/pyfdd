
#import sys
#sys.path.append('/home/eric/PycharmProjects/CustomWidgets/')
#sys.path.append('/home/eric/ericathome/home/eric/PycharmProjects/CustomWidgets')
#print(sys.path)

from pyfdd.core.datapattern.CustomWidgets import AngleMeasure
from matplotlib.widgets import RectangleSelector


import numpy as np
import numpy.ma as ma
import os
import bisect as bis
import scipy.ndimage
import math
import matplotlib as ml
import matplotlib.pyplot as plt
import struct
import warnings
import json, io
import copy


def create_detector_mesh(n_h_pixels, n_v_pixels, pixel_size, distance):
    #same as in PyFDD/patterncrator
    """
    create a mesh for the detector.
    returns a xmesh matrix with the angular values of the detector in the horizontal axis
    and a ymesh matrix with the angular values of the detector in the vertical axis
    all distances must have the same units
    :param n_h_pixels: number of horizontal pixels
    :param n_v_pixels: number of vertical pixels
    :param pixel_size: size of pixel
    :param distance: distance from detector to sample
    :return: x and y mesh matrixes
    """
    d_theta = np.arctan(pixel_size/distance) * 180 / np.pi
    x_i = 0.5 * (n_h_pixels-1) * d_theta
    y_i = 0.5 * (n_v_pixels-1) * d_theta
    x = np.arange(n_h_pixels) * d_theta - x_i
    y = np.arange(n_v_pixels) * d_theta - y_i
    xmesh, ymesh = np.meshgrid(x,y)
    return xmesh, ymesh


class MpxHist:
    """
    Class to hold some useful methods for dealing with histograms in Medipix program
    """
    def __init__(self, values):
        if isinstance(values, ma.MaskedArray):
            values = values[~values.mask]
        self.hist, self.bin_edges = np.histogram(values.reshape((1, values.size)), bins=5000)
        self.normalized_integral = self.hist.cumsum()/float(self.hist.sum())
        self.mean = np.mean(values.reshape((1, values.size)))
        self.std = np.std(values.reshape((1, values.size)))

    def get_bins_from_percentiles(self, percentiles):
        lowbin = bis.bisect(self.normalized_integral, percentiles[0], lo=1, hi=len(self.normalized_integral))-1
            # having lo=1 ensures lowbin is never -1
        highbin = bis.bisect(self.normalized_integral, percentiles[1])

        # I decided to round the ticks because in previous versions the tick label would write all decimal cases
        # It seems that now its ok without rounding but I will still round at the 4th case, instead of the 1st. v0.7.0
        if self.bin_edges[lowbin] != 0.:
            # low bin precision defined as 4 - order of mag
            p10_low = 10**(4 - int(math.floor(math.log10(abs(self.bin_edges[lowbin])))))
            # floor in precision point
            lowtick = np.floor(self.bin_edges[lowbin] * p10_low) / p10_low #self.bin_edges[lowbin]
        else:
            lowtick = 0
        # high bin precision defined as 4 - order of mag
        p10_high = 10**(4 - int(math.floor(math.log10(abs(self.bin_edges[highbin])))))
        hightick = np.ceil(self.bin_edges[highbin] * p10_high) / p10_high #self.bin_edges[highbin]
        return lowtick, hightick

    def get_percentiles_from_ticks(self, ticks):
        # bin_edges is 1 longer than the normalized integral
        lowbin = bis.bisect(self.bin_edges[:-2], ticks[0], lo=0, hi=len(self.bin_edges[:-2]))
        # having lo=1 ensures lowbin is never -1
        highbin = bis.bisect(self.bin_edges[:-2], ticks[1], lo=1, hi=len(self.bin_edges[:-2]))
        low_percentile = np.round(self.normalized_integral[lowbin], 2)
        high_percentile = np.round(self.normalized_integral[highbin], 2)
        percentiles = (low_percentile, high_percentile)
        return percentiles



class DataPattern:
    """
    A class to hold a data pattern and methods for
    - Matrix manipulation (manip_);
    - IO (io_);
    - Angular calibration (ang_);
    - Draw (draw_).
    """
    def __init__(self, file_path=None, pattern_array=None, verbose = 1,  **kwargs):
        if file_path is None and pattern_array is None:
            raise ValueError('Please input a file path or a pattern array')
        if not file_path is None and not pattern_array is None:
            raise ValueError('Please input a file path or a pattern array')

        self.verbose = verbose

        # real size of pixels between chips
        self.real_size = kwargs.get('real_size', 1)
        self.nChipsX = kwargs.get('nChipsX', 1)
        self.nChipsY = kwargs.get('nChipsY', 1)

        assert isinstance(self.real_size,int), 'real_size should be int'
        assert isinstance(self.nChipsX, int), 'nChipsX should be int'
        assert isinstance(self.nChipsY, int), 'nChipsY should be int'

        # initialization of values for IO
        self.filename_in = ''
        self.path_in = ''
        self.filetype_in = ''

        # create data objects for detector mesh
        self.is_mesh_defined = False
        self.xmesh = np.array([[]])
        self.ymesh = np.array([[]])
        (self.ny, self.nx) = (None, None)

        # values for angular calibration
        self.pixel_size_mm = None
        self.distance = None
        self.reverse_x = False

        # values for manipulation methods
        self.mask_central_pixels = 0
        self.rm_edge_pixels = 0

        # orientation variables
        self.center = (0,0)
        self.angle = 0

        # importing matrix
        if not pattern_array is None:
            self.matrixOriginal = ma.array(data=pattern_array.copy(),mask=False)
            (self.ny, self.nx) = self.matrixOriginal.shape
        elif not file_path is None:
            if os.path.isfile(file_path):
                self._io_load(file_path)
            else:
                raise IOError('File does not exist: %s' % file_path)
        self.matrixCurrent = self.matrixOriginal.copy()
        self.matrixDrawable = self.matrixOriginal.copy()

        # creating mesh
        # ang_range = self.ang_get_range(self.distance, self.matrixDrawable.shape[0] * self.pixel_size_mm)
        # self.range = kwargs.get('range', (-ang_range / 2.0, ang_range / 2.0))
        if not self.is_mesh_defined:
            self.manip_create_mesh()

        # inicialization medipix histogram
        self.hist = MpxHist(self.matrixCurrent)

        # Draw variables
        self.ax = None
        self.ang_wid = None
        self.rectangle_limits = None
        self.RS = None

    def __have_same_attributes(self, other):

        # verify if possible and get values
        assert isinstance(other, DataPattern), "Add object is not a DataPattern"

        # check if the shape is the same
        if not self.matrixCurrent.shape == other.matrixCurrent.shape:
            raise ValueError("error the medipix matrices have different shape")

        # check if the number of chips is the same
        if not (self.nChipsX == other.nChipsX and
                        self.nChipsY == other.nChipsY):
            raise ValueError("error, the DataPattern have different number of chips")

        # check if the real size of central pixels is the same
        if not (self.real_size == other.real_size):
            raise ValueError("error, the DataPattern have different real size of central pixels")

        # check if the mesh is the same
        if self.is_mesh_defined is True and other.is_mesh_defined is True:
            if not (np.allclose(self.xmesh, other.xmesh) and
                    np.allclose(self.ymesh, other.ymesh)):
                raise ValueError("error, the DataPattern have different angular mesh")

    def __add__(self, other):

        self.__have_same_attributes(other)

        new_pattern = self.matrixCurrent.data + other.matrixCurrent.data
        new_pattern_mask = self.matrixCurrent.mask + other.matrixCurrent.mask

        new_mm = self.copy()

        new_mm.matrixOriginal = ma.array(data=new_pattern.copy(), mask=new_pattern_mask.copy())
        new_mm.matrixCurrent = ma.array(data=new_pattern.copy(), mask=new_pattern_mask.copy())

        return new_mm

    def __sub__(self, other):
        self.__have_same_attributes(other)

        new_pattern = self.matrixCurrent.data - other.matrixCurrent.data
        new_pattern_mask = self.matrixCurrent.mask + other.matrixCurrent.mask

        new_mm = self.copy()

        new_mm.matrixOriginal = ma.array(data=new_pattern.copy(), mask=new_pattern_mask.copy())
        new_mm.matrixCurrent = ma.array(data=new_pattern.copy(), mask=new_pattern_mask.copy())

        return new_mm

    def __rmul__(self, other):

        return self.__mul__(other)


    def __mul__(self, other):

        # other needs to be a float
        other = np.float(other)

        new_pattern = ma.masked_array(data=self.matrixCurrent.data * other, mask=self.matrixCurrent.mask)

        # Create new MM
        new_mm = self.copy()

        new_mm.matrixOriginal = new_pattern.copy()
        new_mm.matrixCurrent = new_pattern.copy()

        return new_mm

    def __rdiv__(self, other):
        # other needs to be a float
        other = np.float(other)

        if other == 0:
            raise ValueError('Dividing by zero.')

        new_pattern = ma.masked_array(data=self.matrixCurrent.data / other, mask=self.matrixCurrent.mask)

        # Create new MM
        new_mm = self.copy()

        new_mm.matrixOriginal = copy.deepcopy(new_pattern)
        new_mm.matrixCurrent = copy.deepcopy(new_pattern)

        return new_mm

    def copy(self):
        # Draw variables should not be copied
        temp_ax = self.ax
        temp_ang_wid = self.ang_wid
        temp_rectangle_limits= self.rectangle_limits
        temp_RS = self.RS

        self.ax = None
        self.ang_wid = None
        self.rectangle_limits = None
        self.RS = None

        new_dp = copy.deepcopy(self)

        self.ax = temp_ax
        self.ang_wid = temp_ang_wid
        self.rectangle_limits = temp_rectangle_limits
        self.RS = temp_RS

        return new_dp

    def get_matrix(self):
        return self.matrixCurrent.copy()

    def get_xymesh(self):
        return self.xmesh.copy(), self.ymesh.copy()

    def set_xymesh(self, xmesh, ymesh):

        if xmesh.shape != ymesh.shape:
            raise ValueError('xmesh and ymesh need to have the same shape')

        if self.ny is None and self.nx is None:
            (self.ny, self.nx) = xmesh.shape

        if (xmesh.shape != (self.ny, self.nx) or
            ymesh.shape != (self.ny, self.nx)):
            raise ValueError('mesh needs to be of shape (ny,nx)')

        self.is_mesh_defined = True
        self.xmesh = np.array(xmesh)
        self.ymesh = np.array(ymesh)

    # ===== - IO Methods - =====
    def _io_load(self, filename):
        """
        understand what is the filetype and saves the file in given format

        :param filename:  is the name or full path to the file
        :return:
        """
        if not os.path.isfile(filename):
            print('Error file not valid.')
            return
        (self.path_in, self.filename_in) = os.path.split(filename)
        (name, self.filetype_in) = os.path.splitext(self.filename_in)

        if self.filetype_in == '.txt':
            self._io_load_ascii()
        elif self.filetype_in == '.2db':
            self._io_load_origin()
        elif self.filetype_in == '.json':
            self.io_load_json()
        else:
            raise ValueError('Unknown requested file type extension')

    def _io_load_ascii(self):
        """
        loads an ascii file containing a matrix
        """
        self.matrixOriginal = ma.array(data=np.loadtxt(os.path.join(self.path_in, self.filename_in)), mask=False)
        (self.ny, self.nx) = self.matrixOriginal.shape

    def io_save_ascii(self, filename, ignore_mask=False, number_format='int'):
        """
        saves the current matrix to an ascii file
        :param filename: is the name or full path to the file
        :param ignore_mask: if True data values are saved instead of a zero where the mask is on
        :param number_format: set the number format, it can be set to 'int' or 'float'
        :return:
        """
        if ignore_mask:
            matrix = self.matrixCurrent.data
        else:
            matrix = self.matrixCurrent.filled(0)

        if number_format == 'int':
            np.savetxt(filename, matrix, "%d")
        if number_format == 'float':
            np.savetxt(filename, matrix, "%f")

    def _io_load_origin(self):
        """
        loads a 2db file containing a matrix
        """
        fileContent = str('')
        with open(os.path.join(self.path_in, self.filename_in), mode='rb') as file:  # b is important -> binary
            self.fileContent = file.read()

        short_sz = 2
        float_sz = 4
        self.nx = struct.unpack("<h", self.fileContent[0:0+short_sz])[0]
        self.ny = struct.unpack("<h", self.fileContent[2:2+short_sz])[0]
        type = struct.unpack("?", bytes(self.fileContent[4:5]))[0]
        temp = struct.unpack(self.ny*self.nx*"f", self.fileContent[5:5+self.ny*self.nx*float_sz])
        self.matrixOriginal = ma.array(data=np.array(temp).reshape((self.ny,self.nx)), mask=False)

    def io_save_origin(self, filename, ignore_mask=False):
        """
        saves the current matrix to an 2db file
        :param filename: is the name or full path to the file
        :param ignore_mask: if True data values are saved instead of a zero where the mask is on
        :return:
        """
        #matrix = DataPattern.manip_correct_central_pix(self.matrixCurrent, self.nChipsX, self.nChipsY, real_size=real_size)
        if ignore_mask:
            matrix = self.matrixCurrent.data
        else:
            matrix = self.matrixCurrent.filled(0)

        (ny, nx) = matrix.shape
        #print(nx,ny)
        with open(filename, mode='wb') as newfile:  # b is important -> binary
            newfile.write(struct.pack("<h", nx))
            newfile.write(struct.pack("<h", ny))
            newfile.write(struct.pack("<?", False))
            tempbytes = struct.pack("<"+ ny * nx * "f", *matrix.reshape((ny * nx)))
            newfile.write(tempbytes)

    def io_save_json(self, jsonfile):
        js_out = {}
        js_out['matrix'] = {}
        js_out['matrix']['data'] = self.matrixCurrent.data.tolist()
        js_out['matrix']['mask'] = self.matrixCurrent.mask.tolist()
        js_out['matrix']['fill_value'] = self.matrixCurrent.fill_value.tolist()
        # real size of pixels between chips
        js_out['real_size'] = self.real_size
        js_out['nchipsx'] = self.nChipsX
        js_out['nchipsy'] = self.nChipsY
        # data objects for detector mesh
        js_out['is_mesh_defined'] = self.is_mesh_defined
        js_out['xmesh'] = self.xmesh.tolist()
        js_out['ymesh'] = self.ymesh.tolist()
        js_out['ny'] = self.ny
        js_out['nx'] = self.nx
        # values for angular calibration
        js_out['pixel_size_mm'] = self.pixel_size_mm
        js_out['distance'] = self.distance
        # values for manipulation methods
        js_out['mask_central_pixels'] = self.mask_central_pixels
        js_out['rm_edge_pixels'] = self.rm_edge_pixels
        # orientation variables
        js_out['center'] = self.center
        js_out['angle'] = self.angle

        # save to file
        #with io.open(jsonfile, 'w', encoding='utf-8') as f:
        #    f.write(str(json.dumps(js_out, ensure_ascii=False, sort_keys=True)))
        with open(jsonfile, 'w') as fp:
            json.dump(js_out, fp)

    def io_load_json(self):
        with open(os.path.join(self.path_in, self.filename_in), mode='r') as fp:
            json_in = json.load(fp)
            matrix_data = json_in['matrix']['data']
            matrix_mask = json_in['matrix']['mask']
            matrix_fill = json_in['matrix']['fill_value']
            self.matrixOriginal = ma.array(data=matrix_data, mask=matrix_mask, fill_value=matrix_fill)
            # real size of pixels between chips
            self.real_size = json_in['real_size']
            self.nChipsX = json_in['nchipsx']
            self.nChipsY = json_in['nchipsy']
            # data objects for detector mesh
            self.is_mesh_defined = json_in['is_mesh_defined']
            self.xmesh = np.array(json_in['xmesh'])
            self.ymesh = np.array(json_in['ymesh'])
            self.ny = json_in['ny']
            self.nx = json_in['nx']
            # values for angular calibration
            self.pixel_size_mm = json_in['pixel_size_mm']
            self.distance = json_in['distance']
            # values for manipulation methods
            self.mask_central_pixels = json_in['mask_central_pixels']
            self.rm_edge_pixels = json_in['rm_edge_pixels']
            # orientation variables
            self.center = json_in['center']
            self.angle = json_in['angle']

    # ===== - Mask Methods - =====

    def load_mask(self, filename, expand_by=0):
        mask = np.loadtxt(filename)
        if mask.shape != self.matrixCurrent.shape:
            raise ValueError('Shape of mask in file does not match the shape of DataPattern')
        mask = self._expand_any_mask(mask, expand_by)
        self.matrixCurrent.mask = (mask == 0)

    def set_mask(self, mask, expand_by=0):
        mask = np.array(mask)
        if mask.shape != self.matrixCurrent.shape:
            raise ValueError('Shape of mask does not match the shape of DataPattern')
        mask = self._expand_any_mask(mask, expand_by)
        self.matrixCurrent.mask = mask

    def save_mask(self, filename):
        np.savetxt(filename, self.matrixCurrent.mask == 0, fmt='%i')

    def expand_mask(self, expand_by=0):
        self.matrixCurrent.mask = self._expand_any_mask(self.matrixCurrent.mask, expand_by)

    def _expand_any_mask(self, mask, expand_by=0):
        '''
        masks pixels that are adjacent to masked pixels up to a distance of expand_by.
        :param mask:
        :param expand_by:
        :return:
        '''

        if not isinstance(expand_by, int):
            raise ValueError('expand_by must be an int')

        kernel = np.ones((expand_by * 2 + 1))
        nr, nc = mask.shape
        new_mask = mask.copy()
        for r in range(nr):
            new_mask[r, :] = np.convolve(new_mask[r, :], kernel, 'same')

        for c in range(nc):
            new_mask[:, c] = np.convolve(new_mask[:, c], kernel, 'same')
        return new_mask

    def set_fit_region(self, distance=2.9, center=None, angle=None):
        if center is None:
            center = self.center
        if angle is None:
            angle = self.angle

        if len(center) != 2:
            raise ValueError('center must be of length 2.')

        angle = angle * np.pi / 180
        v1 = np.array([np.cos(angle), np.sin(angle)])
        v2 = np.array([np.sin(angle), -np.cos(angle)])

        xy = np.stack([self.xmesh - center[0], self.ymesh - center[1]], axis=-1)

        distance1 = np.abs(np.dot(xy, v1))
        distance2 = np.abs(np.dot(xy, v2))

        if distance >= 0:
            condition = ((distance1 > distance) | (distance2 > distance))
        elif distance < 0:
            condition = ~((distance1 > -distance) | (distance2 > -distance))

        self.matrixCurrent = ma.masked_where(condition, self.matrixCurrent)

    def mask_std(self, std=6, expand_by=0):
        hist = MpxHist(self.matrixCurrent)
        condition = ((self.matrixCurrent <= hist.mean - std * hist.std) | \
                     (self.matrixCurrent >= hist.mean + std * hist.std))
        mask = self._expand_any_mask(condition, expand_by)
        self.matrixCurrent = ma.masked_where(mask==1, self.matrixCurrent)

    def clear_mask(self):
        self.matrixCurrent.mask = False

    # ===== - Matrix Manipulation Methods - =====

    def undo_all(self):
        self.matrixCurrent = self.matrixOriginal.copy()
        self.matrixDrawable = self.matrixOriginal.copy()

    def manip_orient(self, strg):
        assert isinstance(strg, str)
        temp_matrix = self.matrixCurrent
        for cmd in strg.lower().split(','):
            if cmd == 'rl':
                # rotate left
                temp_matrix = np.rot90(temp_matrix)
            elif cmd == 'rr':
                # rotate right
                temp_matrix = np.rot90(temp_matrix, 3)
            elif cmd == 'mv':
                # vertical mirror
                temp_matrix = np.flipud(temp_matrix)
            elif cmd == 'mh':
                # horizontal mirror
                temp_matrix = np.fliplr(temp_matrix)
            else:
                print('Orientation command not understood')
        self.matrixCurrent = temp_matrix

    def manip_smooth(self, fwhm, matrix='Current'):
        #print("smoothing" , fwhm)
        gauss_smooth = fwhm/2.32
        if matrix == 'Current':
            self.matrixCurrent.data[:,:] = scipy.ndimage.gaussian_filter(self.matrixCurrent.data, gauss_smooth)  #, mode='nearest')
        elif matrix == 'Drawable':
            self.matrixDrawable.data[:,:] = scipy.ndimage.gaussian_filter(self.matrixDrawable.data, gauss_smooth)

    def manip_correct_central_pix(self):
        if self.real_size <= 1 and (self.nChipsX > 1 or self.nChipsY > 1):
            warnings.warn('The value for the central pixel real size is set to ', self.real_size)
        nx = self.matrixCurrent.shape[0] + (2 * self.real_size - 2) * (self.nChipsX - 1)
        ny = self.matrixCurrent.shape[0] + (2 * self.real_size - 2) * (self.nChipsY - 1)
        # temp_matrix = -np.ones((ny, nx))
        temp_matrix1 = np.zeros((self.matrixCurrent.shape[0], nx))
        temp_matrix2 = np.zeros((ny, nx))
        mask_update1 = np.ones((self.matrixCurrent.shape[0], nx))==1
        mask_update2 = np.ones((ny, nx))==1

        for interX in range(0, self.nChipsX):
            dock_i = interX * (256 + 2 * self.real_size - 2)
            dock_f = dock_i + 256
            temp_matrix1[:, dock_i:dock_f] = self.matrixCurrent.data[:, interX*256:interX*256 + 256]
            mask_update1[:, dock_i:dock_f] = self.matrixCurrent.mask[:, interX*256:interX*256 + 256]

        for interY in range(0, self.nChipsY):
            dock_i = interY * (256 + 2 * self.real_size - 2)
            dock_f = dock_i + 256
            temp_matrix2[dock_i:dock_f, :] = temp_matrix1[interY*256:interY*256 + 256, :]
            mask_update2[dock_i:dock_f, :] = mask_update1[interY*256:interY*256 + 256, :]

        self.matrixCurrent = ma.array(data=temp_matrix2,mask=mask_update2)
        # Update mesh
        self.manip_create_mesh()

    def zero_central_pix(self, rm_central_pix=None):
        rm_central_pix = int(rm_central_pix)
        if rm_central_pix is not None:
            self.rm_central_pix = rm_central_pix
        #print('Number of chips - ', self.nChipsX*self.nChipsY)
        (ny, nx) = self.matrixCurrent.shape
        xstep = nx // self.nChipsX
        ystep = nx // self.nChipsY
        # print(xstep, ystep, rm_central_pix)
        for ix in range(self.nChipsX-1):
            self.matrixCurrent.mask[:,xstep-rm_central_pix:xstep+rm_central_pix] = True
        for iy in range(self.nChipsY - 1):
            self.matrixCurrent.mask[ystep - rm_central_pix:ystep + rm_central_pix,:] = True

    def remove_edge_pixel(self, rm_edge_pix=0):
        '''
        This function is used to trim edge pixels
        :param rm_edge_pix: number of edge pixels to remove
        :return:
        '''
        assert isinstance(rm_edge_pix, int), 'number of edge pixels to remove should be int'
        if rm_edge_pix > 0:
            self.matrixCurrent = self.matrixCurrent[rm_edge_pix:-rm_edge_pix, rm_edge_pix:-rm_edge_pix]
            # Update mesh
            self.xmesh = self.xmesh[rm_edge_pix:-rm_edge_pix, rm_edge_pix:-rm_edge_pix]
            self.ymesh = self.ymesh[rm_edge_pix:-rm_edge_pix, rm_edge_pix:-rm_edge_pix]

    def _update_compress_factors(self, factor, rm_central_pix, rm_edge_pix, consider_single_chip):

        (ny, nx) = self.matrixCurrent.shape

        # Quad detector compression
        if (2 == self.nChipsX and 2 == self.nChipsY) and consider_single_chip is False:
            # verify if zeroed central pixels are divided by factor
            if 0 <= rm_central_pix:
                # Ensure that the division rest is zero
                central_gap = (2 * (self.real_size + rm_central_pix - 1))
                rest = central_gap % factor
                if rest != 0:
                    rm_central_pix += (factor - rest) / 2
                    warnings.warn("warning removed central pixels increased to " + str(rm_central_pix) +
                                  ", rest is " + str(rest))

            # verify if the rest of the matrix is divisable by factor
            chip_size = 256 # size of a single timepix chip

            if ny != self.nChipsY * (chip_size + self.real_size - 1) or \
               nx != self.nChipsX * (chip_size + self.real_size - 1):
                warnings.warn('Compression of a quad chip assumes a chip size of 256 pixels')

            rest = (chip_size - rm_edge_pix - rm_central_pix) % factor
            if rest != 0:
                rm_edge_pix += rest
                warnings.warn("warning removed edge pixels increased to " + str(rm_edge_pix))

        # Compression of a single chip
        elif (1 == self.nChipsX and 1 == self.nChipsY) or consider_single_chip is True:
            if ny < nx:
                n_min = ny
                n_min_name = 'ny'
            else:
                n_min = nx
                n_min_name = 'nx'

            # The smallest side sets the rm_edge_pix
            rest = (n_min - 2 * rm_edge_pix) % factor
            if rest != 0:
                rm_edge_pix += rest / 2
                print("warning removed edge pixels increased to ", rm_edge_pix)

            # crop largest side if needed
            if n_min_name == 'ny':
                rest = (nx - 2 * rm_edge_pix) % factor
                print('rest/2', rest/2)
                retrnArr = self.matrixCurrent.data[:, int(np.floor(rest/2)):nx - int(np.ceil(rest/2))]
                retrnMa = self.matrixCurrent.mask[:, int(np.floor(rest/2)):nx - int(np.ceil(rest/2))]
                self.matrixCurrent = ma.array(data=retrnArr, mask=(retrnMa >= 1))

            elif n_min_name == 'nx':
                rest = (ny - 2 * rm_edge_pix) % factor
                retrnArr = self.matrixCurrent.data[int(np.floor(rest/2)):ny - int(np.ceil(rest/2)), :]
                retrnMa = self.matrixCurrent.mask[int(np.floor(rest/2)):ny - int(np.ceil(rest/2)), :]
                self.matrixCurrent = ma.array(data=retrnArr, mask=(retrnMa >= 1))

        return factor, rm_central_pix, rm_edge_pix

    def manip_convert_to_single_chip(self):
        self.nChipsX = 1
        self.nChipsY = 1
        self.real_size = 1

    def manip_compress(self, factor=2, rm_central_pix=0, rm_edge_pix=0, consider_single_chip=False):
        '''
        This function reduces the binning of the matrix in a smart way.
        Removed central pixels are not merged with data bins.
        It expects a matrix which central pixels have already been expanded.
        It will increase removed central or edge pixels if needed to match the factor
        :param factor:
        :return:
        '''
        #TODO update for arbitrary vertical and horizontal size

        # Inicial verifications
        if rm_central_pix is not None:
            self.rm_central_pix = rm_central_pix
        if ((self.nChipsX > 1 or self.nChipsY > 1) and
             self.real_size <= 1):
            warnings.warn('The value for the central pixel real size is set to ' + str(self.real_size))

        assert isinstance(factor,int), 'factor should be int'
        assert isinstance(rm_central_pix, int), 'number of central pixels to remove should be int'
        assert isinstance(rm_edge_pix, int), 'number of edge pixels to remove should be int'
        assert isinstance(consider_single_chip, bool), 'consider_single_chip should be bool'

        # update factors to ensure matrix is devisable by factor
        factor, rm_central_pix, rm_edge_pix = \
            self._update_compress_factors(factor, rm_central_pix, rm_edge_pix, consider_single_chip)

        # calculate final shape
        (ny, nx) = self.matrixCurrent.shape

        final_size = [int((ny - rm_edge_pix * 2) / factor), int((nx - rm_edge_pix * 2) / factor)]
        if self.verbose >= 1:
            print('final_size', final_size)

        # Update masked central pixels acoordingly
        self.zero_central_pix(rm_central_pix + (self.real_size - 1))

        # Reshaping the matrix
        yslice = slice(int(np.floor(rm_edge_pix)), ny-int(np.ceil(rm_edge_pix)))
        xslice = slice(int(np.floor(rm_edge_pix)), nx-int(np.ceil(rm_edge_pix)))

        retrnArr = self.matrixCurrent.data[yslice, xslice]\
                       .reshape([final_size[0], factor, final_size[1], factor]).sum(3).sum(1)
        retrnMa = self.matrixCurrent.mask[yslice, xslice] \
                        .reshape([final_size[0], factor, final_size[1], factor]).sum(3).sum(1)
        self.matrixCurrent = ma.array(data=retrnArr, mask=(retrnMa>=1))

        # Update mesh
        self.xmesh = self.xmesh[yslice, xslice] \
                        .reshape([final_size[0], factor, final_size[1], factor]).mean(3).mean(1)
        self.ymesh = self.ymesh[yslice, xslice] \
            .reshape([final_size[0], factor, final_size[1], factor]).mean(3).mean(1)
        if self.pixel_size_mm is not None:
            self.pixel_size_mm *= factor
        #self.manip_create_mesh()
        # After compression convert to a single chip
        self.manip_convert_to_single_chip()

    def manip_create_mesh(self, pixel_size=None, distance=None, reverse_x=None):
        if pixel_size is not None:
            self.pixel_size_mm = pixel_size
        if distance is not None:
            self.distance = distance
        if reverse_x is not None:
            assert isinstance(reverse_x,bool), "reverse_x needs to be bool."
            self.reverse_x = reverse_x
        if pixel_size is not None and distance is not None:
            self.xmesh, self.ymesh = create_detector_mesh(self.matrixCurrent.shape[1],self.matrixCurrent.shape[0],
                                                          self.pixel_size_mm, self.distance)
            self.is_mesh_defined = True
        else:
            if self.is_mesh_defined:
                self.xmesh, self.ymesh = create_detector_mesh(self.matrixCurrent.shape[1], self.matrixCurrent.shape[0],
                                                              self.pixel_size_mm, self.distance)
            else:
                # create detector mesh
                xm = np.arange(self.matrixCurrent.shape[1])
                ym = np.arange(self.matrixCurrent.shape[0])
                self.xmesh, self.ymesh = np.meshgrid(xm, ym)
        if self.reverse_x:
            self.xmesh = np.fliplr(self.xmesh)

        # alternative method
        #manip_create_mesh(self, shape, axis_range):
        #x_ang = np.linspace(axis_range[0], axis_range[1], shape[1])
        #y_ang = np.linspace(axis_range[2], axis_range[3], shape[0])
        #X_ang, Y_ang = np.meshgrid(x_ang, y_ang)
        #return X_ang, Y_ang

    # ===== - Angular Calibration Methods - =====
    @staticmethod
    def ang_get_range(distance, side):
        return 2 * math.degrees(math.atan(side / (2 * distance)))

    # ===== - Draw Methods - =====
    def draw(self, axes, blank_masked=True, **kwargs):

        assert isinstance(axes, plt.Axes)
        self.ax = axes
        fig = ml.axis.Axis.get_figure(axes)

        ticks = None
        percentiles = None
        if 'ticks' in kwargs.keys():
            ticks = kwargs.get('ticks')
            if len(ticks) != 2:
                raise ValueError("ticks must be of length 2 with the colormap limits, for example [0.8, 2]")
        else:
            percentiles = kwargs.get('percentiles', [0.00, 1.0])
            if len(percentiles) != 2:
                raise ValueError("percentiles must be of length 2, for example [0.01, 0.99]")

        colormap = kwargs.get('colormap', 'jet')  # PiYG #coolwarm #spectral
        n_color_bins = kwargs.get('n_color_bins', 10)
        smooth_fwhm = kwargs.get('smooth_fwhm', 0)
        plot_type = kwargs.get('plot_type', 'pixels') #pixels or contour
        xlabel = kwargs.get('xlabel', r'x-angle $\theta[°]$')
        ylabel = kwargs.get('ylabel', r'y-angle $\omega[°]$')
        zlabel = kwargs.get('zlabel', 'Counts')
        title = kwargs.get('title', '2D pattern - ' + str(self.matrixCurrent.shape[0]) +
                           'x' + str(self.matrixCurrent.shape[1]))

        if not self.matrixCurrent.shape == self.xmesh.shape or not self.matrixCurrent.shape == self.ymesh.shape:
            print(self.matrixCurrent.shape, self.ymesh.shape)
            raise ValueError('Pattern and mesh shape dont match')


        self.matrixDrawable = self.matrixCurrent.copy()

        if not smooth_fwhm <= 0:
            self.manip_smooth(smooth_fwhm, matrix='Drawable')

        if blank_masked is False:
            self.matrixDrawable.mask = 0
            if self.nChipsX > 1 and self.real_size > 1:
                #TODO This is not valid if the matrix is compressed
                half = (np.array(self.matrixDrawable.shape)/2).astype(np.int)
                #print([half[0]-(self.real_size-1),half[0]-(self.real_size-1)])
                self.matrixDrawable.mask[half[0] - (self.real_size-1):half[0] + (self.real_size-1), :] = True
                self.matrixDrawable.mask[:, half[1] - (self.real_size - 1):half[1] + (self.real_size - 1)] = True

        imgCmap = ml.cm.get_cmap(colormap)
        if ticks is not None:
            lowtick, hightick = ticks
        else:
            lowtick, hightick = self.get_ticks(percentiles)

        if plot_type == 'contour':
            # set up to n_color_bins levels at nice locations
            levels = ml.ticker.MaxNLocator(nbins=n_color_bins).tick_values(lowtick, hightick)
            # set up exactly n_color_bins levels (alternative)
            #levels = ml.ticker.LinearLocator(numticks=n_color_bins+1).tick_values(lowtick, hightick)
            ret = axes.contourf(self.xmesh, self.ymesh, self.matrixDrawable, cmap=imgCmap, levels=levels)
            if self.reverse_x == True:
                axes.invert_xaxis()
        elif plot_type == 'pixels':
            # the extent needs to acount for the last pixel space therefore add the ste size
            xstep = self.xmesh[0, 1] - self.xmesh[0, 0]
            ystep = self.ymesh[1, 0] - self.ymesh[0, 0]
            extent = [self.xmesh[0,0], self.xmesh[0,-1] + xstep, self.ymesh[0,0], self.ymesh[-1,0] + ystep]
            #print('extent - ', extent)
            ret = axes.imshow(self.matrixDrawable, cmap=imgCmap, interpolation='None', aspect='auto',\
                             vmin=lowtick, vmax=hightick, origin='lower', extent=extent)
        else:
            raise ValueError('plot_type not recognized, use contour or pixels')

        cb = fig.colorbar(ret, ax=axes, use_gridspec=True)

        cb.set_label(zlabel)
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.axis('image')

        return axes, cb

    def callonangle(self, center, angle):
        self.center = center
        self.angle = angle
        self.ang_wid = None

    def get_angle_tool(self):
        if self.ax is None:
            raise ValueError('No axes are defined')
        self.center = (0,0)
        self.angle = 0
        self.ang_wid = AngleMeasure(self.ax, self.callonangle)

    def mask_pixel(self, i, j):
        self.matrixCurrent.mask[i, j] = True

    def mask_rectangle(self, rectangle_limits):
        condition = ((self.xmesh <= rectangle_limits[1]) &
                     (self.xmesh >= rectangle_limits[0]) &
                     (self.ymesh <= rectangle_limits[3]) &
                     (self.ymesh >= rectangle_limits[2]))
        self.matrixCurrent = ma.masked_where(condition, self.matrixCurrent)

    def mask_below(self, value):
        condition = self.matrixCurrent <= value
        self.matrixCurrent = ma.masked_where(condition, self.matrixCurrent)

    def mask_above(self, value):
        condition = self.matrixCurrent >= value
        self.matrixCurrent = ma.masked_where(condition, self.matrixCurrent)

    def onselect_RS(self, eclick, erelease):
        'eclick and erelease are matplotlib events at press and release'
        rectangle_limits = np.array([eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata])
        self.RS = None
        self.mask_rectangle(rectangle_limits)

    def get_rectangle_mask_tool(self):
        if self.ax is None:
            raise ValueError('No axes are defined')
        rectprops = dict(facecolor='red', edgecolor='black',
                         alpha=0.8, fill=True)
        # useblit=True is necessary for PyQt
        self.RS = RectangleSelector(self.ax, self.onselect_RS, drawtype='box', useblit=True, interactive=False,
                                    rectprops=rectprops)

    def get_ticks(self, percentiles):
        if len(percentiles) != 2:
            raise ValueError("percentiles must be of length 2, for example [0.01, 0.99]")
        self.hist = MpxHist(self.matrixCurrent)
        lowtick, hightick = self.hist.get_bins_from_percentiles(percentiles)
        return [lowtick, hightick]
