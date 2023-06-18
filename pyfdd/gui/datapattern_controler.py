

import sys
import os
import warnings
import json
import io

from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from PySide2 import QtCore, QtGui, QtWidgets, uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT #as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from pyfdd.core.datapattern.plot_widgets import AngleMeasurement
import matplotlib.pyplot as plt  # do not use pyplot but import it to ensure mpl works
import matplotlib as mpl
import seaborn as sns
import numpy as np

import pyfdd

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from pyfdd.gui.qt_designer.buildmesh_dialog import Ui_BuildMeshDialog
from pyfdd.gui.qt_designer.colorscale_dialog import Ui_ColorScaleDialog
from pyfdd.gui.qt_designer.setlabels_dialog import Ui_SetLabelsDialog
from pyfdd.gui.qt_designer.importsettings_dialog import Ui_ImportSettingsDialog
from pyfdd.gui.qt_designer.editorientation_dialog import Ui_EditOrientationDialog
import pyfdd.gui.config as config

from specific_tools import make_array_from_scatterdata


# Set style
sns.set_style('white')
sns.set_context('talk')


class NavigationToolbar(NavigationToolbar2QT):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom')]

    def __init__(self, canvas, parent_widget, coordinates=False):
        super(NavigationToolbar, self).__init__(canvas=canvas,
                                                parent=parent_widget,
                                                coordinates=coordinates)

        # spacer widget for left
        left_spacer = QtWidgets.QWidget(parent=parent_widget)
        left_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        #left_spacer.setStyleSheet("background-color:black;")
        # spacer widget for right
        right_spacer = QtWidgets.QWidget(parent=parent_widget)
        right_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        #right_spacer.setContentsMargins(0,0,0,0)
        #right_spacer.setStyleSheet("background-color:black;")
        #self.setStyleSheet("background-color:white;")

        # Insert spacer before home button
        self.insertWidget(self._actions['home'], left_spacer)
        #NavigationToolbar2QT._init_toolbar(self)
        self.addWidget(right_spacer)



class SetLabels_dialog(QtWidgets.QDialog, Ui_SetLabelsDialog):
    def __init__(self, parent_widget, dp_controler):
        super(SetLabels_dialog, self).__init__(parent_widget)
        self.parent_widget = parent_widget
        self.dp_controler = dp_controler
        self.setupUi(self)

        default_labels_suggestions = {'title': 'Channeling Pattern',
                              'xlabel': r'x-angle $\theta[°]$',
                              'ylabel': r'y-angle $\omega[°]$',
                              'zlabel': 'Counts'}

        labels_suggestions = default_labels_suggestions if not \
            config.parser.has_option('datapattern', 'labels_suggestions') else \
            config.getdict('datapattern', 'labels_suggestions')

        self.new_labels = dict()

        for key in labels_suggestions.keys():
            if self.dp_controler.plot_labels[key] == '':
                self.new_labels[key] = labels_suggestions[key]
            else:
                self.new_labels[key] = self.dp_controler.plot_labels[key]

        self._init_le_string()

        # Connect signals
        self.le_title.editingFinished.connect(self.call_le_title)
        self.le_x_axis.editingFinished.connect(self.call_le_x_axis)
        self.le_y_axis.editingFinished.connect(self.call_le_y_axis)
        self.le_z_axis.editingFinished.connect(self.call_le_z_axis)
        self.accepted.connect(self.update_config)

    def _init_le_string(self):
        """
        Instatiate the initial values of the line edit boxes
        :return:
        """
        self.le_title.setText(self.new_labels['title'])
        self.le_x_axis.setText(self.new_labels['xlabel'])
        self.le_y_axis.setText(self.new_labels['ylabel'])
        self.le_z_axis.setText(self.new_labels['zlabel'])

    def update_config(self):
        # update config
        config.parser['datapattern']['labels_suggestions'] = json.dumps(self.new_labels)

    def call_le_title(self):
        self.new_labels['title'] = self.le_title.text()

    def call_le_x_axis(self):
        self.new_labels['xlabel'] = self.le_x_axis.text()

    def call_le_y_axis(self):
        self.new_labels['ylabel'] = self.le_y_axis.text()

    def call_le_z_axis(self):
        self.new_labels['zlabel'] = self.le_z_axis.text()

    def get_settings(self):
        return self.new_labels


class ColorScale_dialog(QtWidgets.QDialog, Ui_ColorScaleDialog):
    def __init__(self, parent_widget, dp_controler):
        super(ColorScale_dialog, self).__init__(parent_widget)
        self.parent_widget = parent_widget
        self.dp_controler = dp_controler
        self.setupUi(self)

        # Set initial values
        self.sb_min_percentile.setValue(self.dp_controler.percentiles[0]*100)
        self.sb_min_tick.setValue(self.dp_controler.ticks[0])
        self.sb_max_percentile.setValue(self.dp_controler.percentiles[1]*100)
        self.sb_max_tick.setValue(self.dp_controler.ticks[1])

        # Init and fill combo boxes
        self.init_cb_colorbar(self.dp_controler.colormap)
        self.init_cb_plot_type(self.dp_controler.plot_type)

        # Set a timer to update ticks from percentiles
        self.qtimer = QtCore.QTimer(parent=self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(self.update_ticks_from_percentiles)

        # Connect signals
        self.sb_min_percentile.valueChanged.connect(self.call_sb_min_percentile)
        self.sb_min_tick.valueChanged.connect(self.call_sb_min_tick)
        self.sb_max_percentile.valueChanged.connect(self.call_sb_max_percentile)
        self.sb_max_tick.valueChanged.connect(self.call_sb_max_tick)
        self.cb_plot_type.currentIndexChanged.connect(self.call_cb_plot_type)
        self.cb_colormap.currentIndexChanged.connect(self.call_cb_colormap)
        self.accepted.connect(self.update_config)

    def init_cb_colorbar(self, current=''):

        colormaps = [
            'jet', 'jet_r', 'Spectral', 'coolwarm', 'coolwarm_r',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'bwr', 'seismic']

        self.cb_colormap.addItems(colormaps)

        if current in colormaps:
            index = colormaps.index(current)
        else:
            index = 0

        self.cb_colormap.setCurrentIndex(index)

    def init_cb_plot_type(self, current=''):

        plot_types = ['pixels', 'contour']

        if current in plot_types:
            index = plot_types.index(current)
        else:
            index = 0

        self.cb_plot_type.setCurrentIndex(index)

    def update_config(self):
        # update config
        config.parser['datapattern']['default_percentiles'] = json.dumps(self.dp_controler.percentiles)
        config.parser['datapattern']['default_plot_type'] = json.dumps(self.dp_controler.plot_type)
        config.parser['datapattern']['default_colormap'] = json.dumps(self.dp_controler.colormap)

    def update_plot(self):
        self.dp_controler.draw_datapattern()
        self.dp_controler.update_infotext()

    def update_ticks_from_percentiles_timer(self):
        # This timer is needed in order to only update the plot after the color scale edit is finished.
        # self.qtimer.stop()  # start automaticaly resets the timer
        self.qtimer.start(100)

    def update_ticks_from_percentiles(self):

        self.dp_controler.ticks = self.dp_controler.dp_plotter.get_ticks(self.dp_controler.percentiles)
        self.sb_min_tick.setValue(self.dp_controler.ticks[0])
        self.sb_max_tick.setValue(self.dp_controler.ticks[1])
        self.update_plot()

    def call_sb_min_percentile(self, value):
        self.dp_controler.percentiles[0] = value / 100
        self.update_ticks_from_percentiles_timer()

    def call_sb_max_percentile(self, value):
        self.dp_controler.percentiles[1] = value / 100
        self.update_ticks_from_percentiles_timer()

    def call_sb_min_tick(self, value):
        self.dp_controler.ticks[0] = value
        self.update_plot()

    def call_sb_max_tick(self, value):
        self.dp_controler.ticks[1] = value
        self.update_plot()

    def call_cb_plot_type(self, index):
        self.dp_controler.plot_type = self.cb_plot_type.itemText(index)
        self.update_plot()

    def call_cb_colormap(self, index):
        self.dp_controler.colormap = self.cb_colormap.itemText(index)
        self.update_plot()


class BuildMesh_dialog(QtWidgets.QDialog, Ui_BuildMeshDialog):
    def __init__(self, parent_widget, dp_controler=None, config_section='datapattern', npixels_active=False):
        super(BuildMesh_dialog, self).__init__(parent_widget)
        self.parent_widget = parent_widget
        self.dp_controler = dp_controler
        self.setupUi(self)

        self.npixels_active = npixels_active
        self.config_section = config_section  # example: 'datapattern', 'patterncreator'

        default_mesh_settings = {'horizontal pixels': 256,
                                 'vertical pixels': 256,
                                 'pixel size': 0.055,
                                 'distance': 315.0,
                                 'angular step': 0.1,
                                 'selected': 'detector'}

        self.mesh_settings = default_mesh_settings if not \
            config.parser.has_option(self.config_section, 'mesh_settings') else \
            config.getdict(self.config_section, 'mesh_settings')

        datapattern = self.dp_controler.get_datapattern() if self.dp_controler is not None else None
        if datapattern is not None:
            self.mesh_settings['horizontal pixels'] = datapattern.nx
            self.mesh_settings['vertical pixels'] = datapattern.ny

        self.load_mesh_settings()

        # Connect signals
        self.le_pixelsize.textEdited.connect(self.set_detector_config)
        self.le_distance.textEdited.connect(self.set_detector_config)
        self.le_angular_step.textEdited.connect(self.set_step_config)
        self.accepted.connect(self.update_config)

    def update_config(self):
        # update config
        config.parser[self.config_section]['mesh_settings'] = json.dumps(self.get_settings())

    def load_mesh_settings(self):

        # Enable or disable number of pixels line edit
        self.le_horizontal_pixels.setEnabled(self.npixels_active)
        self.le_vertical_pixels.setEnabled(self.npixels_active)

        self.le_horizontal_pixels.setText(str(self.mesh_settings['horizontal pixels']))
        self.le_vertical_pixels.setText(str(self.mesh_settings['vertical pixels']))
        self.le_pixelsize.setText(str(self.mesh_settings['pixel size']))
        self.le_distance.setText(str(self.mesh_settings['distance']))
        self.le_angular_step.setText(str(self.mesh_settings['angular step']))
        if self.mesh_settings['selected'] == 'detector':
            self.rb_detector.setChecked(True)
            self.rb_step.setChecked(False)
        elif self.mesh_settings['selected'] == 'step':
            self.rb_detector.setChecked(False)
            self.rb_step.setChecked(True)
        else:  # default to detector
            self.mesh_settings['selected'] = 'detector'
            self.rb_detector.setChecked(True)
            self.rb_step.setChecked(False)

    def get_settings(self):
        settings = dict()
        settings['horizontal pixels'] = int(self.le_horizontal_pixels.text())
        settings['vertical pixels'] = int(self.le_vertical_pixels.text())
        settings['pixel size'] = float(self.le_pixelsize.text())
        settings['distance'] = float(self.le_distance.text())
        settings['angular step'] = float(self.le_angular_step.text())
        if self.rb_detector.isChecked() and not self.rb_step.isChecked():
            settings['selected'] = 'detector'
        elif not self.rb_detector.isChecked() and self.rb_step.isChecked():
            settings['selected'] = 'step'
        else:
            warnings.warn('One and only one option must be selected')
        return settings

    def set_detector_config(self):
        self.rb_detector.setChecked(True)
        self.rb_step.setChecked(False)

    def set_step_config(self):
        self.rb_detector.setChecked(False)
        self.rb_step.setChecked(True)


class ImportSettings_dialog(QtWidgets.QDialog, Ui_ImportSettingsDialog):
    def __init__(self, parent_widget):
        super(ImportSettings_dialog, self).__init__(parent_widget)
        self.parent_widget = parent_widget
        self.setupUi(self)

        dummy_configuration = {'label': 'name',
                               'detector type': 'single',
                               'orientation': ''}

        self.default_config_labels = ['Single chip', 'Pad6 EC-Sli', 'Tpx-quad EC-Sli', 'New configuration']
        self.default_import_config = [dummy_configuration.copy(),
                                      dummy_configuration.copy(),
                                      dummy_configuration.copy(),
                                      dummy_configuration.copy()]
        # Single chip
        self.default_import_config[0]['label'] = self.default_config_labels[0]
        # Pad EC-Sli
        self.default_import_config[1]['label'] = self.default_config_labels[1]
        self.default_import_config[1]['detector type'] = 'single'
        self.default_import_config[1]['orientation'] = 'cc'
        # Tpx quad
        self.default_import_config[2]['label'] = self.default_config_labels[2]
        self.default_import_config[2]['detector type'] = 'quad'
        self.default_import_config[2]['orientation'] = 'cc, mh'
        # New configuration
        self.default_import_config[3]['label'] = self.default_config_labels[3]

        self.import_labels = self.default_config_labels.copy() if not \
            config.parser.has_option('datapattern', 'import_labels') else \
            config.getlist('datapattern', 'import_labels')

        self.import_config = self.default_import_config.copy() if not \
            config.parser.has_option('datapattern', 'import_config') else \
            config.getdict('datapattern', 'import_config')

        self.selected_import = 0 if not \
            config.parser.has_option('datapattern', 'selected_import') else \
            config.getdict('datapattern', 'selected_import')

        self.load_import_config()
        self.refresh_editables()

        # connect signalsload_import_config
        self.cb_import_config.currentIndexChanged.connect(self.refresh_editables)
        self.pb_delete_configuration.clicked.connect(self.delete_entry)
        self.accepted.connect(self.update_config)

        # set regular expression validator
        reg_ex = QtCore.QRegExp(
            r'^(\s*(cw|cc|mh|mv)\s*)(,\s*(cw|cc|mh|mv)\s*)*$')  # accepts commands like cw,cc,mh,mv
        input_validator = QtGui.QRegExpValidator(reg_ex, parent=self.le_orientation_commands)
        self.le_orientation_commands.setValidator(input_validator)

    def load_import_config(self):
        """
        Load the configurations into the widget
        :return:
        """

        # Checks
        current_keys = [entry['label'] for entry in self.import_config]
        # Ensure labels and configs the same length
        if len(self.import_labels) != len(self.import_config):
            print(1)
            self.load_defaut_config()
            return
        # Ensure the default single chip is there
        if (self.import_labels[0] != 'Single chip') or\
            ('Single chip' not in current_keys):
            print(2)
            self.load_defaut_config()
            return
        # Ensure a new configuration is possible
        if (self.import_labels[-1] != 'New configuration') or\
            ('New configuration' not in current_keys):
            print(3)
            print(current_keys)
            print(self.import_labels[-1])
            self.load_defaut_config()
            return

        # Set combo box
        self.cb_import_config.clear()
        self.cb_import_config.addItems(self.import_labels)
        self.cb_import_config.setCurrentIndex(self.selected_import)

    def load_defaut_config(self):
        """
        If there is a problem with the configuration from the .ini file the defaults are loaded
        :return:
        """
        warnings.warn('There was a problem with stored import configurations. Reseting to defaults.')
        self.import_labels = self.default_config_labels
        self.import_config = self.default_import_config
        self.selected_import = 0

        self.load_import_config()

    def refresh_editables(self):
        """
        Refresh the editables with the right values and enable them if the 'New configuration' label is chosen.
        :return:
        """
        # configuration keys {'label', 'detector type', 'orientation'}

        self.selected_import = self.cb_import_config.currentIndex()

        # Enable or disable all widgets
        # Single chip state
        if self.import_labels[self.selected_import] == 'Single chip' and \
                self.selected_import == 0:
            self.w_editables.setEnabled(False)
            self.pb_delete_configuration.setEnabled(False)
        # New configuration state
        elif self.import_labels[self.selected_import] == 'New configuration' and \
                self.selected_import == len(self.import_labels)-1:
            self.w_editables.setEnabled(True)
            self.pb_delete_configuration.setEnabled(False)
        # User configurations
        else:
            self.w_editables.setEnabled(False)
            self.pb_delete_configuration.setEnabled(True)

        # Configuration label
        self.le_config_label.setText(self.import_config[self.selected_import]['label'])

        # Detector type radio buttons
        if self.import_config[self.selected_import]['detector type'] == 'single':
            self.rb_single_chip.setChecked(True)
            self.rb_timepix_quad.setChecked(False)
        elif self.import_config[self.selected_import]['detector type'] == 'quad':
            self.rb_single_chip.setChecked(False)
            self.rb_timepix_quad.setChecked(True)
        else:
            # default to single
            self.rb_single_chip.setChecked(True)
            self.rb_timepix_quad.setChecked(False)

        # Orientation commands
        self.le_orientation_commands.setText(self.import_config[self.selected_import]['orientation'])

    def update_config(self):
        """
        update configs and register new config if needed
        :return:
        """

        if self.selected_import == len(self.import_labels)-1:
            # New configuration
            new_config = {'label': self.le_config_label.text(),
                          'detector type': 'single' if self.rb_single_chip.isChecked() else 'quad',
                          'orientation': self.le_orientation_commands.text()}

            # Ensure only one label named 'New configuration'
            if new_config['label'] == 'New configuration':
                i = 1
                while new_config['label'] in self.import_labels:
                    new_config['label'] = 'New configuration ({})'.format(i)
                    i += 1

            # If label exists overide
            if new_config['label'] in self.import_labels:
                idx = self.import_labels.index(new_config['label'])
                self.import_config[idx] = new_config.copy()
                self.selected_import = idx
            # Else add to last position
            else:
                self.import_labels.insert(-1, new_config['label'])
                self.import_config.insert(-1, new_config.copy())
                self.selected_import = len(self.import_labels) - 2 # one before the last

        # update config
        config.parser['datapattern']['import_labels'] = json.dumps(self.import_labels)
        config.parser['datapattern']['import_config'] = json.dumps(self.import_config)
        config.parser['datapattern']['selected_import'] = json.dumps(self.selected_import)

    def delete_entry(self):
        """
        Delete selected configuration entry
        :return:
        """
        idx = self.selected_import
        self.import_labels.pop(idx)
        self.import_config.pop(idx)
        # Reset combo box
        self.cb_import_config.clear()
        self.cb_import_config.addItems(self.import_labels)
        self.cb_import_config.setCurrentIndex(0)

    def get_settings(self):
        """
        Get settings from the selected configuration
        :return:
        """
        return self.import_config[self.selected_import].copy()

class EditOrientation_dialog(QtWidgets.QDialog, Ui_EditOrientationDialog):
    """ Class to edit the orientation settings."""
    def __init__(self, parent_widget, dp_controler):
        super(EditOrientation_dialog, self).__init__(parent_widget)
        self.setupUi(self)

        if not isinstance(dp_controler, DataPatternControler):
            raise ValueError('Parrent widget needs to be of the DataPatternControler type.')

        self.parent_widget = parent_widget
        self.dp_controler = dp_controler

        x, y = self.dp_controler.datapattern.center
        phi = self.dp_controler.datapattern.angle

        self.le_xyphi.setText(f'({x:.2f}, {y:.2f}, {phi:.1f})')

    def get_settings(self):
        text = self.le_xyphi.text()
        values = [float(x) for x in text.strip('()').split(',')]
        x = values[0]
        y = values[1]
        phi = values[2]

        return x, y, phi


class DataPatternControler(QtCore.QObject):
    """ Data pattern controler class"""

    datapattern_opened = QtCore.pyqtSignal()  # A single DP file is opened. Excludes add or import.
    datapattern_changed = QtCore.pyqtSignal()  # The DP is changed
    datapattern_saved = QtCore.pyqtSignal()  # The DP saved

    def __init__(self, parent_widget=None):

        super(DataPatternControler, self).__init__()

        self.parent_widget = parent_widget
        #self.mainwindow = self.parent_widget.mainwindow
        #self.infotext = infotext_box
        self.mainwindow = None
        #self.infotext = None

        # Set config section
        if not config.parser.has_section('datapattern'):
            config.parser.add_section('datapattern')

        default_percentiles = [0.05, 0.99] if not config.parser.has_option('datapattern', 'default_percentiles') else \
            config.getlist('datapattern', 'default_percentiles')
        default_plot_type = 'pixels' if not config.parser.has_option('datapattern', 'default_plot_type') else \
            config.getlist('datapattern', 'default_plot_type')
        default_colormap = 'jet' if not config.parser.has_option('datapattern', 'default_colormap') else \
            config.getlist('datapattern', 'default_colormap')

        # initiate variables
        self.datapattern = None
        self.dp_plotter = None
        self.percentiles = default_percentiles
        self.plot_type = default_plot_type
        self.colormap = default_colormap
        self.ticks = None
        self.changes_saved = True

        # mpl variables
        #self.mpl_layout = None
        self.mpl_canvas = None
        self.mpl_toolbar = None
        self.maskpixel_mpl_cid = None
        self.cursor_enter_mpl_cid = None
        self.cursor_exit_mpl_cid = None
        self.rselect_mpl = None
        self.plot_labels = {'title': '',
                            'xlabel': '',
                            'ylabel': '',
                            'zlabel': ''}
        self.ang_wid = None

        # Set up matplotlib canvas

        # self.pltfig = plt.figure() # don't use pyplot
        self.pltfig = mpl.figure.Figure()
        #gs = self.pltfig.add_gridspec(1, 1, left=0.15, right=0.85, top=0.9, bottom=0.2)
        gs = self.pltfig.add_gridspec(1, 1, left=0.11, right=0.95, top=0.92, bottom=0.15)
        self.plot_ax = self.pltfig.add_subplot(gs[0])
        #self.plot_ax = self.pltfig.add_subplot(111)
        self.plot_ax.set_aspect('equal')
        self.colorbar_ax = None
        ## self.addmpl()
        # call tight_layout after addmpl
        #self.pltfig.tight_layout()

        # Connect signals
        # Changed the DP saved status
        self.datapattern_opened.connect(lambda: self._set_saved(True))
        self.datapattern_changed.connect(lambda: self._set_saved(False))
        self.datapattern_saved.connect(lambda: self._set_saved(True))

    def set_widgets_and_layouts(self, mpl_layout=None, mainwindow=None, infotext_box=None):
        if mainwindow is not None:
            self.mainwindow = mainwindow
        if infotext_box is not None:
            self.infotext = infotext_box
        self.addmpl(mpl_layout)

    def _set_saved(self, is_saved: bool):
        """
        Used to register if the current DP is saved or not.
        :param is_saved:
        :return:
        """
        self.changes_saved = is_saved

    def are_changes_saved(self):
        """
        Is the current DP saved?.
        :return: True if the DP has been saved.
        """
        return self.changes_saved

    def addmpl(self, mpl_layout):

        def clipboard_handler(event):
            print('event.key', event.key)
            if event.key == 'ctrl+c':
                self.call_copy_to_clipboard()

        if mpl_layout is not None:
            self.mpl_layout = mpl_layout
        else:
            print('mpl_layout is None')
            return

        # get background color from widget and convert it to RBG
        if self.mainwindow is not None:
            pyqt_bkg = self.parent_widget.mainwindow.palette().color(QtGui.QPalette.Background).getRgbF()
            mpl_bkg = mpl.colors.rgb2hex(pyqt_bkg)
            self.pltfig.set_facecolor(mpl_bkg)

        self.mpl_canvas = FigureCanvas(self.pltfig)
        self.mpl_layout.addWidget(self.mpl_canvas)
        self.mpl_canvas.draw()

        self.mpl_toolbar = NavigationToolbar(self.mpl_canvas,
                                             self.parent_widget, coordinates=False)
        self.mpl_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.mpl_layout.addWidget(self.mpl_toolbar)

        # connect status bar coordinates display
        self.mpl_canvas.mpl_connect('motion_notify_event', self.on_move)
        self.mpl_canvas.mpl_connect('key_press_event', clipboard_handler)

    def refresh_mpl_color(self, new_mpl_bkg=None):
        # get background color from widget and convert it to RBG
        if new_mpl_bkg is None:
            pyqt_bkg = self.mpl_toolbar.palette().color(QtGui.QPalette.Background).getRgbF()
            mpl_bkg = mpl.colors.rgb2hex(pyqt_bkg)
        else:
            mpl_bkg = new_mpl_bkg
        self.pltfig.set_facecolor(mpl_bkg)

    def set_datapattern(self, datapattern):
        if not isinstance(datapattern, pyfdd.DataPattern):
            raise ValueError('input was not of the pyfdd.DataPattern type')
        self.datapattern = datapattern

        # Draw pattern and update info text
        self.draw_new_datapattern()
        self.update_infotext()
        self.datapattern_changed.emit()

    def get_datapattern(self) -> (pyfdd.DataPattern, None):
        if self.datapattern is not None:
            return self.datapattern.copy()
        else:
            return None

    def set_dp_filename(self, filename: str):
        # empty virtual method
        pass

    def open_dp_call(self):
        """
        Open a json datapattern file
        :return:
        """
        open_path = '' if not config.parser.has_option('datapattern', 'open_path') else \
            config.get('datapattern', 'open_path')
        filename = QtWidgets.QFileDialog.getOpenFileName(self.parent_widget,
                                                         'Open DataPattern',
                                                         directory=open_path,
                                                         filter='DataPattern (*.json)',
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename == ('', ''):  # Cancel
            return

        try:
            self.datapattern = pyfdd.DataPattern(filename[0])
        except:
            QtWidgets.QMessageBox.warning(self.parent_widget, 'Warning message',
                                          'Error while opening the data file.')
        else:
            # Draw pattern and update info text
            self.draw_new_datapattern()
            self.update_infotext()
            # Give filename to data pattern widget for it to be displayed
            self.set_dp_filename(filename[0])

            # Emit opened signal
            self.datapattern_opened.emit()

            # update config
            open_path = os.path.dirname(filename[0])
            config.parser['datapattern']['open_path'] = open_path

    def openadd_dp_call(self):
        """
        Open a json datapattern file
        :return:
        """
        open_path = '' if not config.parser.has_option('datapattern', 'open_path') else \
            config.get('datapattern', 'open_path')
        filename = QtWidgets.QFileDialog.getOpenFileNames(self.parent_widget,
                                                          'Add DataPatterns',
                                                          directory=open_path,
                                                          filter='DataPattern (*.json)',
                                                          options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename == ([], ''):  # Cancel (first is an empty list)
            return

        try:
            new_datapattern = self.datapattern
            for each in filename[0]:
                if new_datapattern is None:
                    new_datapattern = pyfdd.DataPattern(each)
                else:
                    new_datapattern = new_datapattern + pyfdd.DataPattern(each)
        except:
            QtWidgets.QMessageBox.warning(self.parent_widget, 'Warning message',
                                          'Error while opening the data file.')
        else:
            self.datapattern = new_datapattern.copy()
            # Draw pattern and update info text
            self.draw_new_datapattern()
            self.update_infotext()
            # Give filename to data pattern widget for it to be displayed
            self.set_dp_filename(filename[0][0])  # Only give first file for display

            # Emit opened first then changed signal
            self.datapattern_opened.emit()
            self.datapattern_changed.emit()

            # update config
            open_path = os.path.dirname(filename[0][0])  # Use first file
            config.parser['datapattern']['open_path'] = open_path

    def save_dp_call(self):
        """
        Save the current json file
        :return:
        """
        if not self.datapattern_exits():
            return

        save_path = '' if not config.parser.has_option('datapattern', 'save_path') else \
            config.get('datapattern', 'save_path')

        filename = QtWidgets.QFileDialog.getSaveFileName(self.parent_widget,
                                                         'Save DataPattern',
                                                         directory=save_path,
                                                         filter='DataPattern (*.json)',
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename == ('', ''):  # Cancel
            return

        self.datapattern.io_save_json(filename[0])

        # Give filename to data pattern widget for it to be displayed
        self.parent_widget.set_dp_filename(filename[0])

        # Emit saved signal
        self.datapattern_saved.emit()

        # update config
        save_path = os.path.dirname(filename[0])
        config.parser['datapattern']['save_path'] = save_path

    def processdata_dp_call(self):
        open_path = '' if not config.parser.has_option('datapattern', 'open_path') else \
            config.get('datapattern', 'open_path')
        filenames = QtWidgets.QFileDialog.getOpenFileNames(self.parent_widget,
                                                         'Import matrix file',
                                                         directory=open_path,
                                                         filter='ClusterViewer ScatterData (*)',
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)

        if filenames == ([], ''):  # Cancel (first is an empty list)
            return

        # Process data
        combined_data_array = None
        try:
            for filename in filenames[0]:
                print(filename)
                if filenames[1] == 'ClusterViewer ScatterData (*)':
                    data_array = make_array_from_scatterdata(filename)
                    if combined_data_array is None:
                        combined_data_array = data_array
                    else:
                        combined_data_array += data_array
                else:
                    pass
        except Exception as e:
            QtWidgets.QMessageBox.warning(self.parent_widget, 'Warning message',
                                          'Error while processing the data.\n' + str(e))
            return

        # Get Import settings
        importsettings_dialog = ImportSettings_dialog(parent_widget=self.parent_widget)
        import_config = {}
        if importsettings_dialog.exec_() == QtWidgets.QDialog.Accepted:
            import_config = importsettings_dialog.get_settings()
            # configuration keys {'label', 'detector type', 'orientation'}
        else:  # Canceled
            return

        # Create DataPattern
        try:
            if import_config['detector type'] == 'single':
                self.datapattern = pyfdd.DataPattern(pattern_array=combined_data_array, nChipsX=1, nChipsY=1, real_size=1)
            elif import_config['detector type'] == 'quad':
                self.datapattern = pyfdd.DataPattern(pattern_array=combined_data_array, nChipsX=2, nChipsY=2, real_size=3)
                self.datapattern.manip_correct_central_pix()
            # Orient
            self.datapattern.manip_orient(import_config['orientation'])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self.parent_widget, 'Warning message',
                                          'Error while importing the data.\n' + str(e))
        else:
            # Draw pattern and update info text
            self.draw_new_datapattern()
            self.update_infotext()

            # Give filename to data pattern widget for it to be displayed
            self.set_dp_filename('')

            # Emit opened signal
            self.datapattern_opened.emit()
            self.datapattern_changed.emit()

            # update config
            open_path = os.path.dirname(filename)
            config.parser['datapattern']['open_path'] = open_path

    def import_dp_call(self):
        open_path = '' if not config.parser.has_option('datapattern', 'open_path') else \
            config.get('datapattern', 'open_path')
        filenames = QtWidgets.QFileDialog.getOpenFileNames(self.parent_widget,
                                                          'Import matrix file',
                                                          directory=open_path,
                                                          filter='Import matrix (*.txt *.csv *.2db)',
                                                          options=QtWidgets.QFileDialog.DontUseNativeDialog)

        if filenames == ([], ''):  # Cancel (first is an empty list)
            return

        importsettings_dialog = ImportSettings_dialog(parent_widget=self.parent_widget)

        import_config = {}
        if importsettings_dialog.exec_() == QtWidgets.QDialog.Accepted:
            import_config = importsettings_dialog.get_settings()
            # configuration keys {'label', 'detector type', 'orientation'}
        else:  # Canceled
            return

        try:
            combined_datapattern = None
            for filename in filenames[0]:
                if import_config['detector type'] == 'single':
                    datapattern = pyfdd.DataPattern(file_path=filename, nChipsX=1, nChipsY=1, real_size=1)
                elif import_config['detector type'] == 'quad':
                    datapattern = pyfdd.DataPattern(file_path=filename, nChipsX=2, nChipsY=2, real_size=3)
                    datapattern.manip_correct_central_pix()
                # Orient
                datapattern.manip_orient(import_config['orientation'])

                if combined_datapattern is None:
                    combined_datapattern = datapattern
                else:
                    combined_datapattern += datapattern

            # Store the combined datapattern or use it as needed
            self.datapattern = combined_datapattern

        except Exception as e:
            QtWidgets.QMessageBox.warning(self.parent_widget, 'Warning message',
                                          'Error while importing the data.\n' + str(e))
        else:
            # Draw pattern and update info text
            self.draw_new_datapattern()
            self.update_infotext()
            # Give filename to data pattern widget for it to be displayed
            self.set_dp_filename('')

            # Emit opened signal
            self.datapattern_opened.emit()
            self.datapattern_changed.emit()

            # update config
            open_path = os.path.dirname(filenames[0][0])
            config.parser['datapattern']['open_path'] = open_path

    def export_dp_call(self):
        if not self.datapattern_exits():
            return

        save_path = '' if not config.parser.has_option('datapattern', 'save_path') else \
            config.get('datapattern', 'save_path')

        filename = QtWidgets.QFileDialog.getSaveFileName(self.parent_widget,
                                                         'Export DataPattern',
                                                         directory=save_path,
                                                         filter='ASCII (*.txt);;CSV (*.csv);;Binary (*.2db)',
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)

        if filename == ('', ''):  # Cancel
            return

        if filename[1] == 'ASCII (*.txt)':
            self.datapattern.io_save_ascii(filename[0])
        elif filename[1] == 'CSV (*.csv)':
            self.datapattern.io_save_csv(filename[0])
        elif filename[1] == 'Binary (*.2db)':
            self.datapattern.io_save_origin(filename[0])
        else:
            pass

        # update config
        save_path = os.path.dirname(filename[0])
        config.parser['datapattern']['save_path'] = save_path

    def saveasimage_dp_call(self):
        if not self.datapattern_exits():
            return

        save_path = '' if not config.parser.has_option('datapattern', 'save_path') else \
            config.get('datapattern', 'save_path')

        filename = QtWidgets.QFileDialog. \
            getSaveFileName(self.parent_widget,
                            'Export DataPattern',
                            directory=save_path,
                            filter='image (*emf *eps *.pdf *.png *.ps *.raw *.rgba *.svg *.svgz)',
                            options=QtWidgets.QFileDialog.DontUseNativeDialog)

        if filename == ('', ''):  # Cancel
            return

        # Save with a white background
        # self.pltfig.set_facecolor('white')
        self.pltfig.savefig(filename[0], dpi=600, facecolor='white')
        # self.pltfig.set_facecolor('#d7d6d5')
        # self.canvas.draw()

        # update config
        save_path = os.path.dirname(filename[0])
        config.parser['datapattern']['save_path'] = save_path

    def datapattern_exits(self):
        """
        Warn if the datapattern is None
        :return: bool
        """
        if self.datapattern is None:
            QtWidgets.QMessageBox.warning(self.parent_widget, 'Warning message', 'The DataPattern does not exist.')
            return False
        else:
            return True

    def draw_new_datapattern(self):
        self.ticks = None
        self.draw_datapattern()

    def draw_datapattern(self):

        # Clear previous axes and colorbar
        if self.dp_plotter is not None:
            self.dp_plotter.clear_draw()

        self.dp_plotter = pyfdd.DataPatternPlotter(self.datapattern)

        if self.ticks is None:
            self.ticks = self.dp_plotter.get_ticks(self.percentiles)

        self.dp_plotter.draw(self.plot_ax,
                             ticks=self.ticks,
                             colormap=self.colormap,
                             plot_type=self.plot_type,
                             **self.plot_labels)
        self.plot_ax, self.colorbar_ax = self.dp_plotter.get_axes()

        # call a few times to keep the figure from moving
        #self.pltfig.tight_layout()
        #self.pltfig.tight_layout()
        #self.pltfig.tight_layout()
        #self.pltfig.tight_layout()
        #self.pltfig.tight_layout()
        self.mpl_canvas.draw()
        # self.plot_ax.set_aspect('equal')

    def update_infotext(self):
        if not hasattr(self, 'infotext'):
            return
        if self.infotext is None:
            #raise warnings.warn('Info text box is not set')
            return

        base_text = 'Total counts: {:.1f}; Valid: {:.1f}\n' \
                    'Active pixels: {:d}; Masked: {:d}\n' \
                    'Angular range (x, y): {:.1f}, {:.1f}\n' \
                    'Pattern orientation (x, y, phi): {:.2f}, {:.2f}, {:.1f}'

        total_c = self.datapattern.pattern_matrix.data.sum()
        total_c_active = self.datapattern.pattern_matrix.sum()
        masked_p = self.datapattern.pattern_matrix.mask.sum()
        active_p = (~self.datapattern.pattern_matrix.mask).sum()
        xm = self.datapattern.xmesh[0, :]
        ym = self.datapattern.ymesh[:, 0]
        x_range = xm[-1] - xm[0]
        y_range = ym[-1] - ym[0]
        x_orient, y_orient = self.datapattern.center
        phi = self.datapattern.angle

        text = base_text.format(total_c, total_c_active, active_p, masked_p, x_range, y_range, x_orient, y_orient, phi)

        self.infotext.setText(text)

    def on_move(self, event):
        if self.mainwindow is None:
            return

        if event.inaxes == self.plot_ax:
            x, y = event.xdata, event.ydata
            if self.dp_plotter is not None:
                i, j = self.get_index_from_xy(x, y)
                # get value with 2 decimal cases. 2db files don't round properly
                z = self.dp_plotter.matrixDrawable[i, j]
                if isinstance(z, float):
                    z = float('{:.1f}'.format(z))
            else:
                z = 0
            message = '(x,y,z) - ({:.2f},{:.2f},{})'.format(x, y, z)

            # add angle measurement to status bar
            if self.ang_wid is not None:
                _, phi = self.ang_wid.get_values()
                message += f'    # Angle {phi:.1f} °'
        else:
            message = ''

        self.mainwindow.statusBar().showMessage(message)

    def use_crosscursor_in_axes(self, on):
        def exit_axes(event):
            QtGui.QGuiApplication.restoreOverrideCursor()

        def enter_axes(event):
            if event.inaxes == self.plot_ax:
                QtGui.QGuiApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        if on:
            self.cursor_enter_mpl_cid = self.mpl_canvas.mpl_connect("axes_enter_event", enter_axes)
            self.cursor_exit_mpl_cid = self.mpl_canvas.mpl_connect("axes_leave_event", exit_axes)
        else:
            self.mpl_canvas.mpl_disconnect(self.cursor_enter_mpl_cid)
            self.mpl_canvas.mpl_disconnect(self.cursor_exit_mpl_cid)

    def get_index_from_xy(self, x, y):
        xm = self.datapattern.xmesh[0, :]
        ym = self.datapattern.ymesh[:, 0]

        # find the last position where x is >= xm
        j = np.where(xm <= x)[0][-1]  # columns
        i = np.where(ym <= y)[0][-1]  # lines

        return i, j

    def on_maskpixelclick(self, event):
        if event.button == 1:
            x, y = event.xdata, event.ydata
            i, j = self.get_index_from_xy(x, y)
            self.datapattern.mask_pixel(i, j)

            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()
            self.datapattern_changed.emit()

    def call_pb_maskpixel(self, pushbutton):
        self.pb_maskpixel = pushbutton
        if not self.datapattern_exits():
            self.pb_maskpixel.setChecked(False)
            return

        if self.pb_maskpixel.isChecked():
            self.maskpixel_mpl_cid = self.mpl_canvas.mpl_connect('button_press_event', self.on_maskpixelclick)
            self.use_crosscursor_in_axes(True)
        else:
            self.mpl_canvas.mpl_disconnect(self.maskpixel_mpl_cid)
            self.use_crosscursor_in_axes(False)

    def on_rectangleselect(self, eclick, erelease):

        # get x and y steps
        # because the location of each pixel is defined as the botom left corner,
        # the rectangle needs to be made larger by one pixel at the botom left
        xm = self.datapattern.xmesh[0, :]
        ym = self.datapattern.ymesh[:, 0]
        xstep = xm[1] - xm[0]
        ystep = ym[1] - ym[0]
        # eclick and erelease are matplotlib events at press and release
        rectangle_limits = np.array([eclick.xdata - xstep, erelease.xdata, eclick.ydata - ystep, erelease.ydata])
        self.datapattern.mask_rectangle(rectangle_limits)

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()
        self.datapattern_changed.emit()

    def call_pb_maskrectangle(self, pushbutton):
        self.pb_maskrectangle = pushbutton
        if not self.datapattern_exits():
            self.pb_maskrectangle.setChecked(False)
            return

        if self.pb_maskrectangle.isChecked():
            rectprops = dict(facecolor='red', edgecolor='black',
                             alpha=0.8, fill=True)
            self.rselect_mpl = RectangleSelector(self.plot_ax, self.on_rectangleselect, useblit=True,
                                                 interactive=False,
                                                 props=rectprops)
            # need to update canvas for RS to work properly
            self.mpl_canvas.draw()
            self.use_crosscursor_in_axes(True)
        else:
            self.rselect_mpl = None
            self.use_crosscursor_in_axes(False)

    def call_pb_maskbelow(self):
        if not self.datapattern_exits():
            return

        mask_bellow = 0 if not config.parser.has_option('datapattern', 'mask_bellow') else \
            config.getint('datapattern', 'mask_bellow')

        value, ok = QtWidgets.QInputDialog.getInt(self.parent_widget, 'Mask below',
                                                  'Mask pixels whose value is lower than or equal to\t\t\t',
                                                  value=mask_bellow, min=0)
        if ok:
            self.datapattern.mask_below(value)

            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()
            self.datapattern_changed.emit()

            # update config
            config.parser['datapattern']['mask_bellow'] = str(value)

    def call_pb_maskabove(self):
        if not self.datapattern_exits():
            return

        mask_above = 9000 if not config.parser.has_option('datapattern', 'mask_above') else \
            config.getint('datapattern', 'mask_above')

        value, ok = QtWidgets.QInputDialog.getInt(self.parent_widget, 'Mask above',
                                                  'Mask pixels whose value is higher than or equal to\t\t\t',
                                                  value=mask_above, min=0)
        if ok:
            self.datapattern.mask_above(value)
            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()
            self.datapattern_changed.emit()

            # update config
            config.parser['datapattern']['mask_above'] = str(value)

    def call_pb_maskedge(self):
        if not self.datapattern_exits():
            return

        mask_edge = 0 if not config.parser.has_option('datapattern', 'mask_edge') else \
            config.getint('datapattern', 'mask_edge')

        value, ok = QtWidgets.QInputDialog.getInt(self.parent_widget, 'Input value',
                                                  'Number of edge pixels to mask\t\t\t',
                                                  value=mask_edge, min=0)
        if ok:
            #self.datapattern.remove_edge_pixel(value)
            self.datapattern.mask_edge_pixel(value)

            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()
            self.datapattern_changed.emit()

            # update config
            config.parser['datapattern']['mask_edge'] = str(value)

    def call_pb_removeedge(self):
        if not self.datapattern_exits():
            return

        remove_edge = 0 if not config.parser.has_option('datapattern', 'remove_edge') else \
            config.getint('datapattern', 'remove_edge')

        value, ok = QtWidgets.QInputDialog.getInt(self.parent_widget, 'Input value',
                                                  'Number of edge pixels to remove\t\t\t',
                                                  value=remove_edge, min=0)
        if ok:
            self.datapattern.remove_edge_pixel(value)
            #self.datapattern.mask_edge_pixel(value)

            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()
            self.datapattern_changed.emit()

            # update config
            config.parser['datapattern']['remove_edge'] = str(value)

    def call_pb_removecentral(self):
        if not self.datapattern_exits():
            return

        remove_central = 0 if not config.parser.has_option('datapattern', 'remove_central') else \
            config.getint('datapattern', 'remove_central')

        value, ok = QtWidgets.QInputDialog.getInt(self.parent_widget, 'Input value',
                                                  'Number of edge pixels to remove\t\t\t',
                                                  value=remove_central, min=0)
        if ok:
            self.datapattern.zero_central_pix(value)

            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()
            self.datapattern_changed.emit()

            # update config
            config.parser['datapattern']['remove_central'] = str(value)

    def call_pb_expandmask(self):
        if not self.datapattern_exits():
            return

        expand_mask = 0 if not config.parser.has_option('datapattern', 'expand_mask') else \
            config.getint('datapattern', 'expand_mask')

        value, ok = QtWidgets.QInputDialog.getInt(self.parent_widget, 'Input value',
                                                  'Mask pixels adjacent to already masked pixels by \t\t\t',  # 0,0)
                                                  value=expand_mask, min=0)
        if ok:
            self.datapattern.expand_mask(value)

            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()
            self.datapattern_changed.emit()

            # update config
            config.parser['datapattern']['expand_mask'] = str(value)

    def call_pb_clearmask(self):
        if not self.datapattern_exits():
            return

        self.datapattern.clear_mask()

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()
        self.datapattern_changed.emit()

    def call_pb_loadmask(self):
        if not self.datapattern_exits():
            return

        mask_path = '' if not config.parser.has_option('datapattern', 'mask_path') else \
            config.get('datapattern', 'mask_path')

        filename = QtWidgets.QFileDialog.getOpenFileName(self.parent_widget,
                                                         'Open Mask',
                                                         directory=mask_path,
                                                         filter='Mask file (*.txt)',
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename == ('', ''):  # Cancel
            return

        self.datapattern.load_mask(filename[0])

        # update config
        mask_path = os.path.dirname(filename[0])
        config.parser['datapattern']['save_path'] = mask_path

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()
        self.datapattern_changed.emit()

    def call_pb_savemask(self):
        if not self.datapattern_exits():
            return

        mask_path = '' if not config.parser.has_option('datapattern', 'mask_path') else \
            config.get('datapattern', 'mask_path')

        filename = QtWidgets.QFileDialog.getSaveFileName(self.parent_widget,
                                                         'Save Mask',
                                                         directory=mask_path,
                                                         filter='Mask file (*.txt)',
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename == ('', ''):  # Cancel
            return

        self.datapattern.save_mask(filename[0])

        # update config
        mask_path = os.path.dirname(filename[0])
        config.parser['datapattern']['save_path'] = mask_path

    def call_pb_buildmesh(self):
        if not self.datapattern_exits():
            return

        buildmesh_dialog = BuildMesh_dialog(parent_widget=self.parent_widget, dp_controler=self,
                                            config_section='datapattern', npixels_active=False)

        if buildmesh_dialog.exec_() == QtWidgets.QDialog.Accepted:
            output = buildmesh_dialog.get_settings()
            if output['selected'] == 'detector':
                self.datapattern.manip_create_mesh(pixel_size=output['pixel size'],
                                                   distance=output['distance'])
            elif output['selected'] == 'step':
                dummy_distance = 1000
                dummy_pixel_size = np.tan(output['angular step'] * np.pi / 180) * dummy_distance
                self.datapattern.manip_create_mesh(pixel_size=dummy_pixel_size,
                                                   distance=dummy_distance)
            else:
                warnings.warn('Non valid selection')
        else:
            pass
            # print('Cancelled')

        buildmesh_dialog.deleteLater()

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()
        self.datapattern_changed.emit()

    def call_pb_compressmesh(self):
        if not self.datapattern_exits():
            return

        compress_mesh = 2 if not config.parser.has_option('datapattern', 'compress_mesh') else \
            config.getint('datapattern', 'compress_mesh')

        value, ok = QtWidgets.QInputDialog.getInt(self.parent_widget, 'Compress pixel mesh',
                                                  'Number of pixels to add together in each direction\t\t\t\n' \
                                                  '(may cause removal of extra pixels at the edges)',
                                                  value=compress_mesh, min=2)
        if ok:
            self.datapattern.manip_compress(factor=value)
            # Draw pattern and update info text
            self.draw_new_datapattern()
            self.update_infotext()
            self.datapattern_changed.emit()

            # update config
            config.parser['datapattern']['compress_mesh'] = str(value)

    def callonangle(self, center, angle):
        self.datapattern.center = center
        self.datapattern.angle = angle
        self.ang_wid = None
        self.pb_orientchanneling.setChecked(False)
        self.use_crosscursor_in_axes(False)

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()
        self.datapattern_changed.emit()

    def call_pb_orientchanneling(self, pushbutton):
        self.pb_orientchanneling = pushbutton
        if not self.datapattern_exits():
            self.pb_orientchanneling.setChecked(False)
            return

        if self.pb_orientchanneling.isChecked():
            self.ang_wid = AngleMeasurement(self.plot_ax, callonangle=self.callonangle)
            self.use_crosscursor_in_axes(True)
        else:
            self.ang_wid = None
            self.use_crosscursor_in_axes(False)

    def call_pb_editorientation(self):

        if not self.datapattern_exits():
            return

        orientationedit_dialog = EditOrientation_dialog(parent_widget=self.parent_widget, dp_controler=self)

        if orientationedit_dialog.exec_() == QtWidgets.QDialog.Accepted:
            x, y, phi = orientationedit_dialog.get_settings()
            # configuration keys {'label', 'detector type', 'orientation'}
        else:  # Canceled
            return

        self.datapattern.set_pattern_angular_pos(center=(x, y), angle=phi)

        # Draw pattern and update info text
        # self.draw_datapattern()
        self.update_infotext()
        self.datapattern_changed.emit()

    def call_pb_angularfitrange(self):
        if not self.datapattern_exits():
            return

        x_orient, y_orient = self.datapattern.center
        phi = self.datapattern.angle

        angular_fit_range = 2.7 if not config.parser.has_option('datapattern', 'angular_fit_range') else \
            config.getfloat('datapattern', 'angular_fit_range')

        value, ok = QtWidgets.QInputDialog.getDouble(self.parent_widget, 'Set angular fit range',
                                                     'Set a valid angular range around the channeling axis\t\t\t\n' \
                                                     '(x={:.2f}, y={:.2f} ,phi={:.2f})'.format(x_orient, y_orient, phi),
                                                     value=angular_fit_range, min=0)

        if ok:
            self.datapattern.set_fit_region(distance=value)

            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()
            self.datapattern_changed.emit()

            # update config
            config.parser['datapattern']['angular_fit_range'] = str(value)

    def call_pb_colorscale(self):
        if not self.datapattern_exits():
            return

        colorscale_dialog = ColorScale_dialog(parent_widget=self.parent_widget, dp_controler=self)
        colorscale_dialog.show()

    def call_pb_setlabels(self):
        if not self.datapattern_exits():
            return

        setlabels_dialog = SetLabels_dialog(parent_widget=self.parent_widget, dp_controler=self)
        if setlabels_dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.plot_labels = setlabels_dialog.get_settings()
            config.parser['datapattern']['plot_labels'] = json.dumps(self.plot_labels)
            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()
        else:  # Cancelled.
            pass

    def call_copy_to_clipboard(self):
        print('called copy')
        # store the image in a buffer using savefig(), this has the
        # advantage of applying all the default savefig parameters
        # such as background color; those would be ignored if you simply
        # grab the canvas using Qt
        buf = io.BytesIO()
        self.pltfig.savefig(buf, dpi=150, facecolor='white')
        QtWidgets.QApplication.clipboard().setImage(QtGui.QImage.fromData(buf.getvalue()))
        buf.close()
