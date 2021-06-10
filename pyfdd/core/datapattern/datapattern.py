#!/usr/bin/env python3

"""
DataPattern is the class to hold 2D patterns.
"""

# Imports from standard library
import os
import warnings
import struct
import json
import copy
import bisect as bis
import math

# Imports from 3rd party
from packaging import version
import numpy as np
import numpy.ma as ma
import scipy.ndimage
import matplotlib as ml
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Imports from project
import pyfdd
from pyfdd.core.datapattern.CustomWidgets import AngleMeasure


def create_detector_mesh(n_h_pixels, n_v_pixels, pixel_size=None, distance=None, d_theta=None):
    """
    Create a mesh for the detector.
    Returns a xmesh matrix with the angular values of the detector in the horizontal axis
    and a ymesh matrix with the angular values of the detector in the vertical axis
    all distances must have the same units
    :param n_h_pixels: Number of horizontal pixels
    :param n_v_pixels: Number of vertical pixels
    :param pixel_size: Size of pixel
    :param distance: Distance from detector to sample
    :param d_theta: Angular step between pixels
    :return: x and y mesh matrixes
    """
    if d_theta is None and pixel_size is not None and distance is not None:
        d_theta = np.arctan(pixel_size/distance) * 180 / np.pi
    elif d_theta is not None and pixel_size is None and distance is None:
        d_theta = d_theta
    else:
        raise ValueError('Define pixel_size and distance or d_theta.')

    x_i = 0.5 * (n_h_pixels-1) * d_theta
    y_i = 0.5 * (n_v_pixels-1) * d_theta
    x = np.arange(n_h_pixels) * d_theta - x_i
    y = np.arange(n_v_pixels) * d_theta - y_i
    xmesh, ymesh = np.meshgrid(x, y)
    return xmesh, ymesh


class MpxHist:
    """
    Class to hold useful methods for dealing with histograms in the datapattern program
    """
    # TODO make creating a hist and getting tick values faster

    def __init__(self, values):
        """
        Init function for MpxHist.
        :param values: Masked array of values to fill in the histogram.
        """
        # Verify type
        if isinstance(values, ma.MaskedArray):
            values = values[~values.mask]

        # Build histogram, the histogram integral and calculate usefull statistic function
        self.hist, self.bin_edges = np.histogram(values.reshape((1, values.size)), bins=5000)
        self.normalized_integral = self.hist.cumsum()/float(self.hist.sum())
        self.mean = np.mean(values.reshape((1, values.size)))
        self.std = np.std(values.reshape((1, values.size)))

    def get_bins_from_percentiles(self, percentiles):
        """
        Get bin values at the defined percentiles.
        :param percentiles: Array-like with 2 values. Lower and upper percentiles.
        :return: Tuple of leght 2 with the corresponding lower and upper tick values.
        """

        # Calculate bin indexes
        lowbin = bis.bisect(self.normalized_integral, percentiles[0], lo=1, hi=len(self.normalized_integral))-1
        # having lo=1 ensures lowbin is never -1
        highbin = bis.bisect(self.normalized_integral, percentiles[1])

        # I decided to round the ticks because in previous versions the tick label would write all decimal cases
        # It seems that now its ok without rounding but I will still round at the 4th case, instead of the 1st. v0.7.0
        if self.bin_edges[lowbin] != 0.:
            # low bin precision defined as 4 - order of mag
            p10_low = 10**(4 - int(math.floor(math.log10(abs(self.bin_edges[lowbin])))))
            # floor in precision point
            lowtick = np.floor(self.bin_edges[lowbin] * p10_low) / p10_low  # self.bin_edges[lowbin]
        else:
            lowtick = 0

        # high bin precision defined as 4 - order of mag
        p10_high = 10**(4 - int(math.floor(math.log10(abs(self.bin_edges[highbin])))))
        hightick = np.ceil(self.bin_edges[highbin] * p10_high) / p10_high  # self.bin_edges[highbin]

        return lowtick, hightick

    def get_percentiles_from_ticks(self, ticks):
        """
        Get percentiles at the defined tick values
        :param ticks: Array-like with 2 values. Lower and upper ticks.
        :return: Tuple of leght 2 with the corresponding lower and upper percentile values.
        """
        # bin_edges is 1 longer than the normalized integral
        lowbin = bis.bisect(self.bin_edges[:-2], ticks[0], lo=0, hi=len(self.bin_edges[:-2]))
        # having lo=1 ensures lowbin is never -1
        highbin = bis.bisect(self.bin_edges[:-2], ticks[1], lo=1, hi=len(self.bin_edges[:-2]))
        low_percentile = np.round(self.normalized_integral[lowbin], 2)
        high_percentile = np.round(self.normalized_integral[highbin], 2)
        percentiles = (low_percentile, high_percentile)
        return percentiles


class DataPatternPlotter:
    """
    Class for plotting DataPatterns.
    """
    def __init__(self, datapattern):
        """
        Init method for the DataPatternPlotter class
        :param datapattern: A DataPattern object.
        """

        if not isinstance(datapattern, DataPattern):
            raise ValueError('datapattern is not of the DataPattern type.')

        self.datapattern = datapattern  # Reference

        # Draw variables
        self.ax = None
        self.colorbar_ax = None
        self.ang_wid = None
        self.rectangle_limits = None
        self.RS = None
        self.matrixDrawable = np.array([])

    def draw(self, axes, blank_masked=True, **kwargs):
        """
        Draw the DataPattern to the given axes.
        :param axes: Matplotlib axes
        :param blank_masked: If true masked pixels will be blanked.
        :param kwargs: Plotting arguments.
        :return:
        """
        if not isinstance(axes, plt.Axes):
            raise ValueError('axes need to be of type pyplot.Axes')

        self.ax = axes
        fig = ml.axis.Axis.get_figure(self.ax)

        # Define ticks and percentiles
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

        # Plot arguments
        colormap = kwargs.get('colormap', 'jet')  # PiYG #coolwarm #spectral
        n_color_bins = kwargs.get('n_color_bins', 10)
        smooth_fwhm = kwargs.get('smooth_fwhm', 0)
        plot_type = kwargs.get('plot_type', 'pixels')  # pixels or contour
        xlabel = kwargs.get('xlabel', r'x-angle $\theta[°]$')
        ylabel = kwargs.get('ylabel', r'y-angle $\omega[°]$')
        zlabel = kwargs.get('zlabel', 'Counts')
        title = kwargs.get('title', '2D pattern - ' + str(self.datapattern.pattern_matrix.shape[0]) +
                           'x' + str(self.datapattern.pattern_matrix.shape[1]))

        # Verify shape of pattern matrix and mesh shape
        if not self.datapattern.pattern_matrix.shape == self.datapattern.xmesh.shape or \
                not self.datapattern.pattern_matrix.shape == self.datapattern.ymesh.shape:
            raise ValueError('Pattern and mesh shape dont match')

        # Get a working matrix
        self.matrixDrawable = self.datapattern.pattern_matrix.copy()

        if not smooth_fwhm < 0:
            self.manip_smooth(smooth_fwhm)

        if blank_masked is False:
            self.matrixDrawable.mask = 0
            if self.datapattern.nChipsX > 1 and self.datapattern.real_size > 1:
                # TODO This is not valid if the matrix is compressed
                half = (np.array(self.matrixDrawable.shape)/2).astype(np.int)
                #print([half[0]-(self.real_size-1),half[0]-(self.real_size-1)])
                self.matrixDrawable.mask[half[0] - (self.datapattern.real_size-1):half[0] + (self.datapattern.real_size-1), :] = True
                self.matrixDrawable.mask[:, half[1] - (self.datapattern.real_size - 1):half[1] + (self.datapattern.real_size - 1)] = True

        # Colormap
        img_cmap = ml.cm.get_cmap(colormap)
        if ticks is not None:
            lowtick, hightick = ticks
        else:
            lowtick, hightick = self.get_ticks(percentiles)

        if plot_type == 'contour':
            # set up to n_color_bins levels at nice locations
            levels = ml.ticker.MaxNLocator(nbins=n_color_bins).tick_values(lowtick, hightick)
            # set up exactly n_color_bins levels (alternative)
            # levels = ml.ticker.LinearLocator(numticks=n_color_bins+1).tick_values(lowtick, hightick)
            ret = self.ax.contourf(self.datapattern.xmesh, self.datapattern.ymesh, self.matrixDrawable, cmap=img_cmap, levels=levels)
            if self.datapattern.reverse_x is True:
                self.ax.invert_xaxis()
        elif plot_type == 'pixels':
            # the extent needs to acount for the last pixel space therefore add the ste size
            xstep = self.datapattern.xmesh[0, 1] - self.datapattern.xmesh[0, 0]
            ystep = self.datapattern.ymesh[1, 0] - self.datapattern.ymesh[0, 0]
            extent = [self.datapattern.xmesh[0, 0],
                      self.datapattern.xmesh[0, -1] + xstep,
                      self.datapattern.ymesh[0, 0],
                      self.datapattern.ymesh[-1, 0] + ystep]
            ret = self.ax.imshow(self.matrixDrawable, cmap=img_cmap, interpolation='None', aspect='auto',
                                 vmin=lowtick, vmax=hightick, origin='lower', extent=extent)
        else:
            raise ValueError('plot_type not recognized, use contour or pixels')

        self.colorbar_ax = fig.colorbar(ret, ax=self.ax, use_gridspec=True)

        self.colorbar_ax.set_label(zlabel)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.axis('image')

        return self.ax, self.colorbar_ax

    def clear_draw(self):
        """
        Clear previous axes and colorbar.
        :return:
        """
        if self.colorbar_ax is not None:
            fig = ml.axis.Axis.get_figure(self.ax)
            #fig.delaxes(fig.axes[1])
            self.colorbar_ax.remove()
        if self.ax is not None:
            self.ax.clear()
            self.ax.reset_position()

    def get_axes(self):
        """
        Get the axes and colorbar.
        :return:
        """
        if self.ax is None or self.colorbar_ax is None:
            warnings.warn('Axes are not properly set.')

        return self.ax, self.colorbar_ax

    def get_ticks(self, percentiles):
        """
        Get tics for the colorbar.
        :param percentiles: Percentiles to set the max and minimum tick values.
        :return:
        """
        if len(percentiles) != 2:
            raise ValueError("percentiles must be of length 2, for example [0.01, 0.99]")
        hist = MpxHist(self.datapattern.pattern_matrix)
        lowtick, hightick = hist.get_bins_from_percentiles(percentiles)
        return [lowtick, hightick]

    def get_angle_tool(self):
        if self.ax is None:
            raise ValueError('No axes are defined')
        self.datapattern.center = (0, 0)
        self.datapattern.angle = 0
        self.ang_wid = AngleMeasure(self.ax, self.datapattern.set_pattern_angular_pos)

    def onselect_RS(self, eclick, erelease):
        # eclick and erelease are matplotlib events at press and release
        rectangle_limits = np.array([eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata])
        self.RS = None
        self.datapattern.mask_rectangle(rectangle_limits)

    def get_rectangle_mask_tool(self):
        if self.ax is None:
            raise ValueError('No axes are defined')
        rectprops = dict(facecolor='red', edgecolor='black',
                         alpha=0.8, fill=True)
        # useblit=True is necessary for PyQt
        self.RS = RectangleSelector(self.ax, self.onselect_RS, drawtype='box', useblit=True, interactive=False,
                                    rectprops=rectprops)

    def manip_smooth(self, fwhm):
        gauss_smooth = fwhm/2.32
        self.matrixDrawable.data[:, :] = scipy.ndimage.gaussian_filter(self.matrixDrawable.data, gauss_smooth)


class DataPattern:
    """
    A class to hold a data pattern and methods for
    - Matrix manipulation (manip_);
    - IO (io_);
    - Angular calibration (ang_);
    """

    def __init__(self, file_path=None, pattern_array=None, verbose=1,  **kwargs):
        """
        Init method for the DataPattern class.
        :param file_path: Path of a datapattern or array file. Do not use if pattern_array is defined.
        :param pattern_array: A 2D array. Do not use if file_path is defined.
        :param verbose: Verbose level.
        :param kwargs: Timepix quad parameters: real_size, nChipsX and nChipsY.
        """

        if file_path is None and pattern_array is None:
            raise ValueError('Please input a file path or a pattern array')
        if file_path is not None and pattern_array is not None:
            raise ValueError('Please input a file path or a pattern array')

        self.verbose = verbose

        # Variables here defined may need to be included in the JSON save and load methods

        # real size of pixels between chips
        self.real_size = kwargs.get('real_size', 1)
        self.nChipsX = kwargs.get('nChipsX', 1)
        self.nChipsY = kwargs.get('nChipsY', 1)

        # Verifications
        if not isinstance(file_path, str) and file_path is not None:
            raise ValueError('file_path should be string.')
        if not isinstance(pattern_array, (list, tuple, np.ndarray)) and pattern_array is not None:
            raise ValueError('pattern_array should be an array like of 2 dimensions.')
        if not isinstance(self.real_size, int):
            raise ValueError('real_size should be int.')
        if not isinstance(self.nChipsX, int):
            raise ValueError('nChipsX should be int.')
        if not isinstance(self.nChipsY, int):
            raise ValueError('nChipsY should be int')

        # Create data objects for detector mesh
        self.is_mesh_defined = False
        self.xmesh = np.array([[]])
        self.ymesh = np.array([[]])
        (self.ny, self.nx) = (None, None)  # (n rows, n columns)

        # Values for angular calibration
        self.pixel_size_mm = None
        self.distance = None
        self.reverse_x = False

        # Values for manipulation methods
        self.mask_central_pixels = 0
        self.rm_edge_pixels = 0
        self.rm_central_pix = 0

        # Orientation variables
        self.center = (0, 0)
        self.angle = 0

        # Importing matrix
        self.pattern_matrix = ma.array([])
        if pattern_array is not None:
            self.pattern_matrix = ma.array(data=pattern_array.copy(), mask=False)
            (self.ny, self.nx) = self.pattern_matrix.shape
        elif file_path is not None:
            if os.path.isfile(file_path):
                self._io_load(file_path)
            else:
                raise IOError('File does not exist: {}}'.format(file_path))
        # Masks
        self.pixels_mask = np.array(np.zeros((self.ny, self.nx)), dtype=bool)
        self.fitregion_mask = np.array(np.zeros((self.ny, self.nx)), dtype=bool)

        # Create mesh
        if not self.is_mesh_defined:
            self.manip_create_mesh()

    def _have_same_attributes(self, other):
        """
        Check if an other DataPattern object has the same attributes in order to do mathematical operation between them.
        :param other: DataPattern object
        """

        # verify if possible and get values
        if not isinstance(other, DataPattern):
            raise ValueError('Operation failed as object is not a DataPattern.')

        # check if the shape is the same
        if not self.pattern_matrix.shape == other.pattern_matrix.shape:
            raise ValueError('Error, the pattern matrices have different shapes.')

        # check if the number of chips is the same
        if not (self.nChipsX == other.nChipsX and
                self.nChipsY == other.nChipsY):
            raise ValueError('Error, the DataPattern have different number of chips.')

        # check if the real size of central pixels is the same
        if not (self.real_size == other.real_size):
            raise ValueError('Error, the DataPattern have different real size of central pixels.')

        # check if the mesh is the same
        if self.is_mesh_defined is True and other.is_mesh_defined is True:
            if not (np.allclose(self.xmesh, other.xmesh) and
                    np.allclose(self.ymesh, other.ymesh)):
                raise ValueError('Error, the DataPattern have different angular mesh.')

    def __add__(self, other):
        """
        Add two DataPattern objects.
        :param other: DataPattern object.
        :return: DataPattern object.
        """

        # Do verifications
        self._have_same_attributes(other)
        assert isinstance(other, DataPattern)

        # Add matrices
        new_pattern = self.pattern_matrix.data + other.pattern_matrix.data
        # new_pattern_mask = self.pattern_matrix.mask + other.pattern_matrix.mask
        new_pattern_pixel_mask = self.pixels_mask + other.pixels_mask
        new_pattern_fitrange_mask = self.fitregion_mask + other.fitregion_mask

        # Make a new DataPattern with the calculated matrices
        new_mm = self.copy()
        new_mm.pattern_matrix = ma.array(data=new_pattern.copy(), mask=False)
        new_mm.pixels_mask = new_pattern_pixel_mask
        new_mm.fitregion_mask = new_pattern_fitrange_mask
        new_mm._update_mask()
        return new_mm

    def __sub__(self, other):
        """
        Subtract a DataPattern object.
        :param other: DataPattern object.
        :return: DataPattern object.
        """

        # Do verifications
        self._have_same_attributes(other)
        assert isinstance(other, DataPattern)

        # Subtract matrix
        new_pattern = self.pattern_matrix.data - other.pattern_matrix.data
        # new_pattern_mask = self.pattern_matrix.mask + other.pattern_matrix.mask
        new_pattern_pixel_mask = self.pixels_mask + other.pixels_mask
        new_pattern_fitrange_mask = self.fitregion_mask + other.fitregion_mask

        # Make a new DataPattern with the calculated matrices
        new_mm = self.copy()
        new_mm.pattern_matrix = ma.array(data=new_pattern.copy(), mask=False)
        new_mm.pixels_mask = new_pattern_pixel_mask
        new_mm.fitregion_mask = new_pattern_fitrange_mask
        new_mm._update_mask()
        return new_mm

    def __rmul__(self, other):
        """
        Right multiplication by a scalar.
        :param other: A numerical value.
        :return: DataPattern object.
        """

        return self.__mul__(other)

    def __mul__(self, other):
        """
        Left multiplication by a scalar.
        :param other: A numerical value.
        :return: DataPattern object.
        """

        # other needs to be a float
        other = np.float(other)

        # Calculate new pattern
        new_pattern = ma.masked_array(data=self.pattern_matrix.data * other, mask=False)

        # Make a new DataPattern with the calculated matrices
        new_mm = self.copy()
        new_mm.pattern_matrix = new_pattern.copy()
        new_mm._update_mask()
        return new_mm

    def __rdiv__(self, other):
        """
        Right division by a scalar.
        :param other: A numerical value.
        :return: DataPattern object.
        """

        # other needs to be a float
        other = np.float(other)

        if other == 0:
            raise ValueError('Dividing by zero.')

        # Calculate new pattern
        new_pattern = ma.masked_array(data=self.pattern_matrix.data / other, mask=False)

        # Make a new DataPattern with the calculated matrices
        new_mm = self.copy()
        new_mm.pattern_matrix = copy.deepcopy(new_pattern)
        new_mm._update_mask()
        return new_mm

    def copy(self):
        """
        Make a copy of itself.
        :return: DataPattern object.
        """
        new_dp = copy.deepcopy(self)
        return new_dp

    def get_matrix(self):
        """
        Get the pattern matrix.
        :return: Numpy array.
        """
        return self.pattern_matrix.copy()

    def get_xymesh(self):
        """
        Get the X and Y mesh.
        :return: Tuple of (x, y) mesh. Each a Numpy array.
        """
        return self.xmesh.copy(), self.ymesh.copy()

    def set_xymesh(self, xmesh, ymesh):
        """
        Set the X and Y mesh.
        :param xmesh: Two dimmentional array of mesh coordinates.
        :param ymesh: Two dimmentional array of mesh coordinates.
        """

        # Verify if the shapes are the same
        if xmesh.shape != ymesh.shape:
            raise ValueError('xmesh and ymesh need to have the same shape.')

        if self.ny is None and self.nx is None:
            (self.ny, self.nx) = xmesh.shape

        if (xmesh.shape != (self.ny, self.nx) or
                ymesh.shape != (self.ny, self.nx)):
            raise ValueError('Mesh needs to be of shape (ny, nx).')

        self.is_mesh_defined = True
        self.xmesh = np.array(xmesh)
        self.ymesh = np.array(ymesh)

    # ===== - IO Methods - =====
    def _io_load(self, filename):
        """
        Loads the DataPattern in the given file format.
        :param filename: Name or full path to the file.
        """
        if not os.path.isfile(filename):
            print('Error file not valid.')
            return

        # Path operations
        (_, filetype) = os.path.splitext(filename)

        # Call the right load function
        if filetype == '.txt':
            self._io_load_ascii(filename)
        elif filetype == '.2db':
            self._io_load_origin(filename)
        elif filetype == '.json':
            self._io_load_json(filename)
        else:
            raise ValueError('Unknown file type extension.')

    def _io_load_ascii(self, filename):
        """
        Loads an ascii file containing a 2D data matrix.
        :param filename: Name or full path to the file.
        """
        self.pattern_matrix = ma.array(data=np.loadtxt(filename), mask=False)
        (self.ny, self.nx) = self.pattern_matrix.shape

    def io_save_ascii(self, filename, ignore_mask=False, number_format='int'):
        """
        Saves the current pattern matrix to an ascii file.
        :param filename: Name or full path to the file.
        :param ignore_mask: If True data values are saved instead of a zero where the mask is on.
        :param number_format: Set the number format, it can be set to 'int' or 'float'
        :return:
        """
        if ignore_mask:
            matrix = self.pattern_matrix.data
        else:
            matrix = self.pattern_matrix.filled(0)

        if number_format == 'int':
            np.savetxt(filename, matrix, "%d")
        if number_format == 'float':
            np.savetxt(filename, matrix, "%f")

    def _io_load_origin(self, filename):
        """
        Loads a 2db file containing a pattern matrix.
        :param filename: Name or full path to the file.
        """
        with open(filename, mode='rb') as file:  # b is important -> binary
            file_content = file.read()

        # Size of short and float types in bytes
        short_sz = 2
        float_sz = 4

        # 2db file has
        # 1 Short with the number of columns
        # 1 Short with the number of lines
        # 1 Dummy byte
        # nx*ny of Floats
        self.nx = struct.unpack("<h", file_content[0:0+short_sz])[0]
        self.ny = struct.unpack("<h", file_content[2:2+short_sz])[0]
        _ = struct.unpack("?", bytes(file_content[4:5]))[0]  # Dummy byte, named type on FDD
        temp = struct.unpack(self.ny*self.nx*"f", file_content[5:5+self.ny*self.nx*float_sz])
        self.pattern_matrix = ma.array(data=np.array(temp).reshape((self.ny, self.nx)), mask=False)

    def io_save_origin(self, filename, ignore_mask=False):
        """
        Saves the current matrix to an 2db file.
        :param filename: Name or full path to the file.
        :param ignore_mask: If True data values are saved instead of a zero where the mask is on.
        """
        if ignore_mask:
            matrix = self.pattern_matrix.data
        else:
            matrix = self.pattern_matrix.filled(0)

        (ny, nx) = matrix.shape
        with open(filename, mode='wb') as newfile:  # b is important -> binary
            newfile.write(struct.pack("<h", nx))
            newfile.write(struct.pack("<h", ny))
            newfile.write(struct.pack("<?", False))
            tempbytes = struct.pack("<" + ny * nx * "f", *matrix.reshape((ny * nx)))
            newfile.write(tempbytes)

    def io_save_json(self, filename):
        """
        Saves the current DataPattern to a JSON file.
        :param filename: Name or full path to the file.
        """
        js_out = dict()
        js_out['pyfdd version'] = pyfdd.__version__
        js_out['matrix'] = dict()
        js_out['matrix']['data'] = self.pattern_matrix.data.tolist()
        js_out['matrix']['pixels_mask'] = self.pixels_mask.tolist()
        js_out['matrix']['fitregion_mask'] = self.fitregion_mask.tolist()
        js_out['matrix']['fill_value'] = self.pattern_matrix.fill_value.tolist()
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
        with open(filename, 'w') as fp:
            json.dump(js_out, fp)

    def _io_load_json(self, filename):
        """
        Loads a DataPatter from a JSON file.
        :param filename: Name or full path to the file.
        """
        with open(filename, mode='r') as fp:
            json_in = json.load(fp)
            # Data and masks
            matrix_data = json_in['matrix']['data']
            if 'mask' in json_in['matrix']:
                # Changed in version 0.8.0.dev1
                self.pixels_mask = np.array(json_in['matrix']['mask'], dtype=bool)
                self.fitregion_mask = np.array(np.zeros(self.pixels_mask.shape), dtype=bool)
            else:
                self.pixels_mask = np.array(json_in['matrix']['pixels_mask'], dtype=bool)
                self.fitregion_mask = np.array(json_in['matrix']['fitregion_mask'], dtype=bool)
            matrix_fill = json_in['matrix']['fill_value']
            self.pattern_matrix = ma.array(data=matrix_data, mask=False, fill_value=matrix_fill)
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

            # Update mask
            self._update_mask()

    # ===== - Mask Methods - =====
    def load_mask(self, filename, expand_by=0):
        """
        Loads a mask file.
        :param filename: Name or full path to the file.
        :param expand_by: Number of pixels by which to expand each masked pixel.
        """
        mask = np.loadtxt(filename)
        if mask.shape != self.pattern_matrix.shape:
            raise ValueError('Shape of mask in file does not match the shape of DataPattern')
        mask = self._expand_any_mask(mask, expand_by)
        self.pixels_mask = np.array((mask == 0), dtype=bool)
        self._update_mask()

    def set_mask(self, mask, expand_by=0):
        """
        Set the datapattern mask.
        :param mask: 2D array.
        :param expand_by: Number of pixels by which to expand each masked pixel.
        """
        mask = np.array(mask)
        if mask.shape != self.pattern_matrix.shape:
            raise ValueError('Shape of mask does not match the shape of DataPattern')
        mask = self._expand_any_mask(mask, expand_by)
        self.pixels_mask = np.array(mask, dtype=bool)
        self._update_mask()

    def _update_mask(self):
        self.pattern_matrix.mask = self.pixels_mask + self.fitregion_mask

    def save_mask(self, filename):
        """
        Save mask to a text file.
        :param filename: Name or full path to the file.
        """
        np.savetxt(filename, self.pixels_mask == 0, fmt='%i')

    def expand_mask(self, expand_by=0):
        """
        Expand the DataPattern mask.
        :param expand_by: Number of pixels by which to expand each masked pixel.
        :return:
        """
        self.pixels_mask = self._expand_any_mask(self.pixels_mask, expand_by)
        self._update_mask()

    @staticmethod
    def _expand_any_mask(mask, expand_by=0):
        """
        Masks pixels that are adjacent to masked pixels up to a distance of expand_by.
        :param mask: 2D array.
        :param expand_by: Number of pixels by which to expand each masked pixel.
        :return: New mask in a 2D array.
        """
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

    def set_pattern_angular_pos(self, center, angle):
        """
        Set the angular position of the pattern.
        :param center: The (x, y) position of the center.
        :param angle: The angular orientation of the pattern in degrees.
        :return:
        """
        self.center = center
        self.angle = angle

    def set_fit_region(self, distance=2.9, center=None, angle=None):
        """
        Set the valid fit region around the center.
        :param distance: Angular distance that defines the region.
        :param center: (x, y) position of the center.
        :param angle: Angular orientation of the pattern.
        :return:
        """
        if center is None:
            center = self.center
        if angle is None:
            angle = self.angle

        if len(center) != 2:
            raise ValueError('center must be of length 2.')

        # Calculate the distance of each point in the pattern.
        angle = angle * np.pi / 180
        v1 = np.array([np.cos(angle), np.sin(angle)])
        v2 = np.array([np.sin(angle), -np.cos(angle)])

        xy = np.stack([self.xmesh - center[0], self.ymesh - center[1]], axis=-1)

        distance1 = np.abs(np.dot(xy, v1))
        distance2 = np.abs(np.dot(xy, v2))

        # Compare the calculated distance with the distange range
        if distance >= 0:
            condition = ((distance1 > distance) | (distance2 > distance))
        else:  # distance < 0
            condition = ~((distance1 > -distance) | (distance2 > -distance))

        # Mask pixels that are outside of the fit region
        self.fitregion_mask = condition
        self._update_mask()

    def mask_pixel(self, i, j):
        """
        Mask single pixel.
        :param i: Line index.
        :param j: Column index.
        :return:
        """
        self.pixels_mask[i, j] = True
        self._update_mask()

    def mask_rectangle(self, rectangle_limits):
        """
        Mask a rectangular area.
        :param rectangle_limits: (x1, x2, y1, y2) limits of the rectangle to mask.
        :return:
        """
        condition = ((self.xmesh <= rectangle_limits[1]) &
                     (self.xmesh >= rectangle_limits[0]) &
                     (self.ymesh <= rectangle_limits[3]) &
                     (self.ymesh >= rectangle_limits[2]))
        self.pixels_mask = self.pixels_mask + condition
        self._update_mask()

    def mask_below(self, value):
        """
        Mask pixels whose value is bellow a value.
        :param value: Mask value threshold.
        :return:
        """
        condition = self.pattern_matrix <= value
        self.pixels_mask = self.pixels_mask + condition
        self._update_mask()

    def mask_above(self, value):
        """
        Mask pixels whose value is above a value.
        :param value: Mask value threshold.
        :return:
        """
        condition = self.pattern_matrix >= value
        self.pixels_mask = self.pixels_mask + condition
        self._update_mask()

    def mask_std(self, std=6, expand_by=0):
        """
        Mask pixels whose value exeed a certain standard deviation.
        :param std: Standard deviation limit
        :param expand_by: Expand masked pixels.
        :return:
        """
        hist = MpxHist(self.pattern_matrix)
        condition = ((self.pattern_matrix <= hist.mean - std * hist.std) |
                     (self.pattern_matrix >= hist.mean + std * hist.std))
        mask = self._expand_any_mask(condition, expand_by)
        self.pixels_mask = self.pixels_mask + mask
        self._update_mask()

    def clear_mask(self, pixels_mask=True, fitregion_mask=False):
        """
        Clear the mask.
        :return:
        """
        if pixels_mask:
            self.pixels_mask = np.array(np.zeros((self.ny, self.nx)), dtype=bool)
        if fitregion_mask:
            self.fitregion_mask = np.array(np.zeros((self.ny, self.nx)), dtype=bool)
        self._update_mask()

    # ===== - Matrix Manipulation Methods - =====

    def manip_orient(self, command_str):
        """
        Orient pattern matrix by rotation and mirroring
        :param command_str: A string of commands. Use cw, cc, mh, mv to rotate clockwise, rotate counterclockwise,
        mirror horizontaly and mirror verticaly.
        :return:
        """
        if not isinstance(command_str, str):
            raise ValueError('command_str must be a string.')

        temp_matrix = self.pattern_matrix.copy()
        new_nx, new_ny = self.nx, self.ny
        new_xmesh, new_ymesh = self.get_xymesh()
        command_str = command_str.lower().replace(' ', '').strip(',')
        for cmd in command_str.split(','):
            if cmd == 'cw':
                # rotate clockwise
                temp_matrix = np.rot90(temp_matrix, 3)
                temp_xmesh = new_xmesh.copy()
                new_xmesh = np.rot90(new_ymesh, 3)
                new_ymesh = np.rot90(temp_xmesh, 1)
                temp_nx = new_nx
                new_nx = new_ny
                new_ny = temp_nx
            elif cmd == 'cc':
                # rotate counterclockwise
                temp_matrix = np.rot90(temp_matrix)
                temp_xmesh = new_xmesh.copy()
                new_xmesh = np.rot90(new_ymesh, 3)
                new_ymesh = np.rot90(temp_xmesh, 1)
                temp_nx = new_nx
                new_nx = new_ny
                new_ny = temp_nx
            elif cmd == 'mv':
                # vertical mirror
                temp_matrix = np.flipud(temp_matrix)
            elif cmd == 'mh':
                # horizontal mirror
                temp_matrix = np.fliplr(temp_matrix)
            elif cmd == '':
                continue
            else:
                print('Orientation command \'{}\' not understood'.format(cmd))

        self.nx, self.ny = new_nx, new_ny
        self.set_xymesh(new_xmesh, new_ymesh)
        self.pattern_matrix = temp_matrix

    def manip_correct_central_pix(self):
        """
        Correct central pixels in quad detectors.
        :return:
        """
        # Do input verifications
        if self.real_size <= 1 and (self.nChipsX > 1 or self.nChipsY > 1):
            # expect a larger real size
            warnings.warn('The value for the central pixel real size is set to ', self.real_size)

        # Calculate new dimentions
        nx = self.pattern_matrix.shape[0] + (2 * self.real_size - 2) * (self.nChipsX - 1)
        ny = self.pattern_matrix.shape[0] + (2 * self.real_size - 2) * (self.nChipsY - 1)
        temp_matrix1 = np.zeros((self.pattern_matrix.shape[0], nx))
        temp_matrix2 = np.zeros((ny, nx))
        # masks are ones so that new pixels are masked
        pixels_mask_update1 = np.ones((self.pattern_matrix.shape[0], nx)) == 1
        pixels_mask_update2 = np.ones((ny, nx)) == 1
        fitregion_mask_update1 = np.ones((self.pattern_matrix.shape[0], nx)) == 1
        fitregion_mask_update2 = np.ones((ny, nx)) == 1

        # Update pattern matrix
        for interX in range(0, self.nChipsX):
            dock_i = interX * (256 + 2 * self.real_size - 2)
            dock_f = dock_i + 256
            temp_matrix1[:, dock_i:dock_f] = self.pattern_matrix.data[:, interX * 256:interX * 256 + 256]
            pixels_mask_update1[:, dock_i:dock_f] = self.pixels_mask[:, interX * 256:interX * 256 + 256]
            fitregion_mask_update1[:, dock_i:dock_f] = self.fitregion_mask[:, interX * 256:interX * 256 + 256]

        for interY in range(0, self.nChipsY):
            dock_i = interY * (256 + 2 * self.real_size - 2)
            dock_f = dock_i + 256
            temp_matrix2[dock_i:dock_f, :] = temp_matrix1[interY*256:interY*256 + 256, :]
            pixels_mask_update2[dock_i:dock_f, :] = pixels_mask_update1[interY*256:interY*256 + 256, :]
            fitregion_mask_update2[dock_i:dock_f, :] = fitregion_mask_update1[interY * 256:interY * 256 + 256, :]

        self.pattern_matrix = ma.array(data=temp_matrix2, mask=False)
        (self.ny, self.nx) = self.pattern_matrix.shape

        # Update mesh
        self.manip_create_mesh()
        # Update mask
        self.pixels_mask = pixels_mask_update2
        self.fitregion_mask = fitregion_mask_update2
        self._update_mask()

    def zero_central_pix(self, rm_central_pix=None):
        """
        Mask the central pixels for a quad detector.
        :param rm_central_pix:
        :return:
        """
        rm_central_pix = int(rm_central_pix)
        if rm_central_pix is not None:
            self.rm_central_pix = rm_central_pix
        else:
            self.rm_central_pix = 0

        (ny, nx) = self.pattern_matrix.shape
        xstep = nx // self.nChipsX
        ystep = nx // self.nChipsY

        for ix in range(self.nChipsX-1):
            self.pixels_mask[:, xstep - rm_central_pix:xstep + rm_central_pix] = True
        for iy in range(self.nChipsY - 1):
            self.pixels_mask[ystep - rm_central_pix:ystep + rm_central_pix, :] = True
        self._update_mask()

    def remove_edge_pixel(self, rm_edge_pix=0):
        """
        This function is used to trim edge pixels.
        :param rm_edge_pix: Number of edge pixels to remove.
        :return:
        """
        if not isinstance(rm_edge_pix, int):
            raise ValueError('The number of edge pixels to remove should be int.')

        if rm_edge_pix > 0:
            self.pattern_matrix = self.pattern_matrix[rm_edge_pix:-rm_edge_pix, rm_edge_pix:-rm_edge_pix]
            # update mask
            self.pixels_mask = self.pixels_mask[rm_edge_pix:-rm_edge_pix, rm_edge_pix:-rm_edge_pix]
            self.fitregion_mask = self.fitregion_mask[rm_edge_pix:-rm_edge_pix, rm_edge_pix:-rm_edge_pix]
            # Update pattern shape and mask
            self.nx -= 2 * rm_edge_pix
            self.ny -= 2 * rm_edge_pix
            self._update_mask()
            # Update mesh
            self.xmesh = self.xmesh[rm_edge_pix:-rm_edge_pix, rm_edge_pix:-rm_edge_pix]
            self.ymesh = self.ymesh[rm_edge_pix:-rm_edge_pix, rm_edge_pix:-rm_edge_pix]

    def mask_edge_pixel(self, n_edge_pix=0):
        """
        This function is used to mask edge pixels.
        :param n_edge_pix: Number of edge pixels to mask.
        :return:
        """
        if not isinstance(n_edge_pix, int):
            raise ValueError('The number of edge pixels to mask should be int.')

        if n_edge_pix > 0:
            self.pixels_mask[0:n_edge_pix, :] = True
            self.pixels_mask[:, 0:n_edge_pix] = True
            self.pixels_mask[-n_edge_pix:, :] = True
            self.pixels_mask[:, -n_edge_pix:] = True

            self._update_mask()

    def _update_compress_factors(self, factor, rm_central_pix, rm_edge_pix, consider_single_chip):
        """
        Update the rm_central_pix and rm_edge_pix in a smart way for the timepix quad or a single chip detector.
        When a single chip pattern is not square this function will also crop the longuest side so that
        it is divisible by factor.
        :param factor: Number of pixels to add together.
        :param rm_central_pix: Number of central pixels to mask.
        :param rm_edge_pix: Number of edge pixels to trim.
        :param consider_single_chip: Treat the detector as a single chip.
        :return:
        """

        (ny, nx) = (self.ny, self.nx)

        # Quad detector compression
        if (2 == self.nChipsX and 2 == self.nChipsY) and consider_single_chip is False:
            rm_central_pix, rm_edge_pix = \
                self._update_compress_factors_tpx_quad(factor, rm_central_pix, rm_edge_pix)

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
                retrn_arr = self.pattern_matrix.data[:, int(np.floor(rest / 2)):nx - int(np.ceil(rest / 2))]
                retrn_pixels_mask = self.pixels_mask[:, int(np.floor(rest / 2)):nx - int(np.ceil(rest / 2))]
                retrn_fitregion_mask = self.fitregion_mask[:, int(np.floor(rest / 2)):nx - int(np.ceil(rest / 2))]
                self.pattern_matrix = ma.array(data=retrn_arr, mask=False)
                self.pixels_mask = retrn_pixels_mask
                self.fitregion_mask = retrn_fitregion_mask
                self._update_mask()
                # Update nx
                self.nx = nx - 2 * int(np.ceil(rest / 2))

            elif n_min_name == 'nx':
                rest = (ny - 2 * rm_edge_pix) % factor
                retrn_arr = self.pattern_matrix.data[int(np.floor(rest / 2)):ny - int(np.ceil(rest / 2)), :]
                retrn_pixels_mask = self.pixels_mask[int(np.floor(rest / 2)):ny - int(np.ceil(rest / 2)), :]
                retrn_fitregion_mask = self.fitregion_mask[int(np.floor(rest / 2)):ny - int(np.ceil(rest / 2)), :]
                self.pattern_matrix = ma.array(data=retrn_arr, mask=False)
                self.pixels_mask = retrn_pixels_mask
                self.fitregion_mask = retrn_fitregion_mask
                self._update_mask()
                # Update ny
                self.ny = ny - 2 * int(np.ceil(rest / 2))

        return factor, rm_central_pix, rm_edge_pix

    def _update_compress_factors_tpx_quad(self, factor, rm_central_pix, rm_edge_pix):

        (ny, nx) = (self.ny, self.nx)

        if nx != ny:
            raise ValueError('A timepix quad needs to have the same number of horizontal and vertical pixels.')

        # verify if the size of the matrix assumes a chip size of 256 pixels
        chip_size = 256  # size of a single timepix chip

        if ny != self.nChipsY * (chip_size + self.real_size - 1) or \
                nx != self.nChipsX * (chip_size + self.real_size - 1):
            apparent_size = (nx/self.nChipsX) + (1 - self.real_size)
            raise ValueError('Compression of a quad chip assumes a chip size of 256 pixels.\n'
                             'Single chip size seems to be {}'.format(apparent_size))

        # verify if zeroed central pixels are divided by factor
        if 0 <= rm_central_pix:
            # Number of bins to remove from the actual matrix
            rm_central_bins = self.real_size - 1 + rm_central_pix
            # Ensure that the division rest is zero
            central_gap = 2 * rm_central_bins
            rest = central_gap % factor
            if rest != 0:
                if (factor - rest) / 2 == int((factor - rest) / 2):
                    rm_central_pix += int((factor - rest) / 2)
                else:
                    # treat each side separately
                    rest = rm_central_bins % factor
                    rm_central_pix += int(factor - rest)
                #warnings.warn("warning removed central pixels increased to " + str(rm_central_pix) +
                #              ", rest is " + str(rest))

        n_pixel_left = chip_size - rm_central_pix
        rest = (n_pixel_left - rm_edge_pix) % factor
        if rest != 0:
            rm_edge_pix += rest
            #warnings.warn("warning removed edge pixels increased to " + str(rm_edge_pix))

        return rm_central_pix, rm_edge_pix

    def manip_convert_to_single_chip(self):
        """
        Convert the current DataPattern to a single chip.
        :return:
        """
        self.nChipsX = 1
        self.nChipsY = 1
        self.real_size = 1

    def manip_compress(self, factor=2, rm_central_pix=0, rm_edge_pix=0, consider_single_chip=False):
        """
        This function reduces the binning of the matrix in a smart way.
        Removed central pixels are not merged with data bins.
        It expects a matrix which central pixels have already been expanded.
        It will increase removed central or edge pixels if needed to match the factor
        :param factor: Number of pixels to add together.
        :param rm_central_pix: Number of central pixels to mask.
        :param rm_edge_pix: Number of edge pixels to trim.
        :param consider_single_chip: Treat the detector as a single chip.
        :return:
        """
        # TODO update for arbitrary vertical and horizontal size

        # Inicial verifications
        if not isinstance(factor, int):
            raise ValueError('factor should be int.')
        if not isinstance(rm_central_pix, int):
            raise ValueError('number of central pixels to remove should be int.')
        if not isinstance(rm_edge_pix, int):
            raise ValueError('number of edge pixels to remove should be int.')
        if not isinstance(consider_single_chip, bool):
            raise ValueError('consider_single_chip should be bool.')

        if (self.nChipsX > 1 or self.nChipsY > 1) and self.real_size <= 1:
            # Larger real size expected
            warnings.warn('The value for the central pixel real size is set to ' + str(self.real_size))

        if rm_central_pix is not None:
            self.rm_central_pix = rm_central_pix

        # update factors to ensure matrix is devisable by factor
        factor, rm_central_pix, rm_edge_pix = \
            self._update_compress_factors(factor, rm_central_pix, rm_edge_pix, consider_single_chip)

        # calculate final shape
        (ny, nx) = (self.ny, self.nx)

        final_size = [int((ny - rm_edge_pix * 2) / factor), int((nx - rm_edge_pix * 2) / factor)]

        # Update masked central pixels acoordingly
        self.zero_central_pix(rm_central_pix + (self.real_size - 1))

        # Reshaping the matrix
        yslice = slice(int(np.floor(rm_edge_pix)), ny-int(np.ceil(rm_edge_pix)))
        xslice = slice(int(np.floor(rm_edge_pix)), nx-int(np.ceil(rm_edge_pix)))

        retrn_arr = self.pattern_matrix.data[yslice, xslice]\
                        .reshape([final_size[0], factor, final_size[1], factor]).sum(3).sum(1)
        retrn_pixels_mask = self.pixels_mask[yslice, xslice] \
                       .reshape([final_size[0], factor, final_size[1], factor]).sum(3).sum(1)
        retrn_fitregion_mask = self.fitregion_mask[yslice, xslice] \
                       .reshape([final_size[0], factor, final_size[1], factor]).sum(3).sum(1)

        self.pattern_matrix = ma.array(data=retrn_arr, mask=False)
        self.pixels_mask = retrn_pixels_mask
        self.fitregion_mask = retrn_fitregion_mask
        self._update_mask()
        (self.ny, self.nx) = final_size

        # Update shape

        # Update mesh
        self.xmesh = self.xmesh[yslice, xslice] \
                         .reshape([final_size[0], factor, final_size[1], factor]).mean(3).mean(1)
        self.ymesh = self.ymesh[yslice, xslice] \
            .reshape([final_size[0], factor, final_size[1], factor]).mean(3).mean(1)
        if self.pixel_size_mm is not None:
            self.pixel_size_mm *= factor

        # After compression convert to a single chip
        self.manip_convert_to_single_chip()

    def manip_create_mesh(self, pixel_size=None, distance=None, reverse_x=None):
        """
        Creates an angular mesh for the pattern according to the setup geometry.
        :param pixel_size: Real size of a pixel.
        :param distance: Distance from the sample to the detector.
        :param reverse_x: If true the x axis increases from right to left.
        :return:
        """
        # Inputs verifications
        if pixel_size is not None:
            self.pixel_size_mm = pixel_size
        if distance is not None:
            self.distance = distance
        if reverse_x is not None:
            if not isinstance(reverse_x, bool):
                raise ValueError('reverse_x needs to be bool.')
            self.reverse_x = reverse_x

        if pixel_size is not None and distance is not None:
            self.xmesh, self.ymesh = create_detector_mesh(self.pattern_matrix.shape[1], self.pattern_matrix.shape[0],
                                                          self.pixel_size_mm, self.distance)
            self.is_mesh_defined = True
        else:
            if self.is_mesh_defined:
                self.xmesh, self.ymesh = create_detector_mesh(self.pattern_matrix.shape[1],
                                                              self.pattern_matrix.shape[0],
                                                              self.pixel_size_mm,
                                                              self.distance)
            else:
                # create detector mesh
                xm = np.arange(self.pattern_matrix.shape[1])
                ym = np.arange(self.pattern_matrix.shape[0])
                self.xmesh, self.ymesh = np.meshgrid(xm, ym)
        if self.reverse_x:
            self.xmesh = np.fliplr(self.xmesh)

    # ===== - Draw Methods - =====
    def draw(self, axes, **kwargs):
        """
        Calls a DataPatternPlotter and draws the pattern into the given axes.
        :param axes: Matplotlib axes
        :param kwargs: Plotting arguments.
        :return:
        """
        dp_plotter = DataPatternPlotter(datapattern=self)
        dp_plotter.draw(axes, **kwargs)

        return dp_plotter

