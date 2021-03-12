
import sys
import os
import warnings
import json

from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from PySide2 import QtCore, QtGui, QtWidgets, uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT #as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from pyfdd.core.datapattern.CustomWidgets import AngleMeasure
import matplotlib.pyplot as plt  # do not use pyplot but import it to ensure mpl works
import matplotlib as mpl
import seaborn as sns
import numpy as np

import pyfdd

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from pyfdd.gui.qt_designer.datapattern_widget import Ui_DataPatternWidget
from pyfdd.gui.qt_designer.buildmesh_dialog import Ui_BuildMeshDialog
from pyfdd.gui.qt_designer.colorscale_dialog import Ui_ColorScaleDialog
from pyfdd.gui.qt_designer.setlabels_dialog import Ui_SetLabelsDialog
import pyfdd.gui.config as config


# Set style
sns.set_style('white')
sns.set_context('talk')


class NavigationToolbar(NavigationToolbar2QT):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom')]

    def _init_toolbar(self):
        # spacer widget for left
        left_spacer = QtWidgets.QWidget()
        left_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        #left_spacer.setStyleSheet("background-color:black;")
        # spacer widget for right
        right_spacer = QtWidgets.QWidget()
        right_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        #right_spacer.setContentsMargins(0,0,0,0)
        #right_spacer.setStyleSheet("background-color:black;")
        #self.setStyleSheet("background-color:white;")

        self.addWidget(left_spacer)
        NavigationToolbar2QT._init_toolbar(self)
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
            if self.dp_controler.plot_labels[key] is '':
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

        # Connect signals
        self.sb_min_percentile.valueChanged.connect(self.call_sb_min_percentile)
        self.sb_min_tick.valueChanged.connect(self.call_sb_min_tick)
        self.sb_max_percentile.valueChanged.connect(self.call_sb_max_percentile)
        self.sb_max_tick.valueChanged.connect(self.call_sb_max_tick)
        self.accepted.connect(self.update_config)

    def update_config(self):
        # update config
        config.parser['datapattern']['default_percentiles'] = json.dumps(self.dp_controler.percentiles)

    def update_plot(self):
        self.dp_controler.draw_datapattern()
        self.dp_controler.update_infotext()

    def update_ticks_from_percentiles(self):

        self.dp_controler.ticks = self.dp_controler.datapattern.get_ticks(self.dp_controler.percentiles)
        self.sb_min_tick.setValue(self.dp_controler.ticks[0])
        self.sb_max_tick.setValue(self.dp_controler.ticks[1])
        self.update_plot()

    def call_sb_min_percentile(self, value):
        self.dp_controler.percentiles[0] = value / 100
        self.update_ticks_from_percentiles()

    def call_sb_max_percentile(self, value):
        self.dp_controler.percentiles[1] = value / 100
        self.update_ticks_from_percentiles()

    def call_sb_min_tick(self, value):
        self.dp_controler.ticks[0] = value
        self.update_plot()

    def call_sb_max_tick(self, value):
        self.dp_controler.ticks[1] = value
        self.update_plot()


class BuildMesh_dialog(QtWidgets.QDialog, Ui_BuildMeshDialog):
    def __init__(self, parent_widget, dp_controler):
        super(BuildMesh_dialog, self).__init__(parent_widget)
        self.parent_widget = parent_widget
        self.dp_controler = dp_controler
        self.setupUi(self)

        default_mesh_settings = {'pixel size': 0.055,
                                 'distance': 315.0,
                                 'angular step': 0.1,
                                 'selected': 'detector'}

        self.mesh_settings = default_mesh_settings if not \
            config.parser.has_option('datapattern', 'mesh_settings') else \
            config.getdict('datapattern', 'mesh_settings')

        self.load_mesh_settings()

        # Connect signals
        self.le_pixelsize.textEdited.connect(self.set_detector_config)
        self.le_distance.textEdited.connect(self.set_detector_config)
        self.le_angstep.textEdited.connect(self.set_step_config)
        self.accepted.connect(self.update_config)

    def update_config(self):
        # update config
        config.parser['datapattern']['mesh_settings'] = json.dumps(self.get_settings())

    def load_mesh_settings(self):

        self.le_pixelsize.setText(str(self.mesh_settings['pixel size']))
        self.le_distance.setText(str(self.mesh_settings['distance']))
        self.le_angstep.setText(str(self.mesh_settings['angular step']))
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
        settings['pixel size'] = float(self.le_pixelsize.text())
        settings['distance'] = float(self.le_distance.text())
        settings['angular step'] = float(self.le_angstep.text())
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


class DataPattern_window(QtWidgets.QMainWindow):
    """ Class to use the data pattern widget in a separate window"""
    def __init__(self, *args, **kwargs):
        super(DataPattern_window, self).__init__(*args, **kwargs)

        # Setup the window
        self.window_title = "Data Pattern"
        self.setWindowTitle(self.window_title)
        self.statusBar()

        # Set a DataPattern widget as central widget
        self.dp_w = DataPattern_widget(mainwindow=self)
        self.setCentralWidget(self.dp_w)
        self.resize(1150, 670)

        self.dp_w.datapattern_changed_or_saved.connect(self.title_update)

    def set_datapattern(self, datapattern):
        self.dp_w.set_datapattern(datapattern)

    def title_update(self):
        if self.dp_w.are_changes_saved() is False:
            if self.window_title[-1] == "*":
                pass
            else:
                self.window_title = self.window_title + '*'
        else:
            if self.window_title[-1] == "*":
                self.window_title = self.window_title[0:-1]
        self.setWindowTitle(self.window_title)


class DataPattern_widget(QtWidgets.QWidget, Ui_DataPatternWidget):
    """ Data pattern widget class"""

    datapattern_opened = QtCore.pyqtSignal()
    datapattern_changed_or_saved = QtCore.pyqtSignal()

    def __init__(self, *args, mainwindow=None, **kwargs):
        """
        Init method for the data pattern widget
        :param args:
        :param mainwindow: Main window object
        :param kwargs:
        """

        super(DataPattern_widget, self).__init__(*args, **kwargs)

        # Alternative way to load the ui created with PyQt creator
        # uic.loadUi('qt_designer/datapattern_widget.ui', self)

        self.setupUi(self)
        self.mainwindow = mainwindow

        # set the mpl widget background colour
        self.mplwindow.setStyleSheet('background: palette(window);')

        # Instantiate datapattern controler
        self.dpcontroler = DataPatternControler(parent_widget=self, mpl_layout=self.mplvl, infotext_box=self.infotext)
        self.dpcontroler.datapattern_opened.connect(self.datapattern_opened.emit)
        self.dpcontroler.datapattern_changed_or_saved.connect(self.datapattern_changed_or_saved.emit)

        # Create a menubar entry for the datapattern
        self.menubar = self.mainwindow.menuBar()
        self.dp_menu = self.setup_menu()

        # Connect signals
        # Pattern manipulation
        self.pb_buildmesh.clicked.connect(self.dpcontroler.call_pb_buildmesh)
        self.pb_compressmesh.clicked.connect(self.dpcontroler.call_pb_compressmesh)
        self.pb_orientchanneling.clicked.connect(lambda:self.dpcontroler.call_pb_orientchanneling(self.pb_orientchanneling))
        self.pb_fitrange.clicked.connect(self.dpcontroler.call_pb_angularfitrange)

        # Mask signals
        self.pb_maskpixel.clicked.connect(lambda: self.dpcontroler.call_pb_maskpixel(self.pb_maskpixel))
        self.pb_maskrectangle.clicked.connect(lambda: self.dpcontroler.call_pb_maskrectangle(self.pb_maskrectangle))
        self.pb_maskpixel.clicked.connect(self.untoggle_pb_maskrectangle)
        self.pb_maskrectangle.clicked.connect(self.untoggle_pb_maskpixel)
        self.pb_maskbelow.clicked.connect(self.dpcontroler.call_pb_maskbelow)
        self.pb_maskabove.clicked.connect(self.dpcontroler.call_pb_maskabove)
        self.pb_removeedge.clicked.connect(self.dpcontroler.call_pb_removeedge)
        self.pb_removecentral.clicked.connect(self.dpcontroler.call_pb_removecentral)
        self.pb_expandmask.clicked.connect(self.dpcontroler.call_pb_expandmask)
        self.pb_clearmask.clicked.connect(self.dpcontroler.call_pb_clearmask)
        self.pb_loadmask.clicked.connect(self.dpcontroler.call_pb_loadmask)
        self.pb_savemask.clicked.connect(self.dpcontroler.call_pb_savemask)

        # Pattern visualization
        self.pb_colorscale.clicked.connect(self.dpcontroler.call_pb_colorscale)
        self.pb_setlabels.clicked.connect(self.dpcontroler.call_pb_setlabels)

    def setup_menu(self):
        dp_menu = self.menubar.addMenu('&Data Pattern')

        # Open DataPattern
        open_act = QtWidgets.QAction('&Open', self)
        open_act.setStatusTip('Open a DataPattern.json file')
        open_act.triggered.connect(self.dpcontroler.open_dp_call)
        dp_menu.addAction(open_act)

        # Open DataPattern
        openadd_act = QtWidgets.QAction('&Add', self)
        openadd_act.setStatusTip('Add DataPatterns together')
        openadd_act.triggered.connect(self.dpcontroler.openadd_dp_call)
        dp_menu.addAction(openadd_act)

        # Import pattern matrix
        import_act = QtWidgets.QAction('&Import', self)
        import_act.setStatusTip('Import a pattern from an ASCII or .2db file')
        import_act.triggered.connect(self.dpcontroler.import_dp_call)
        dp_menu.addAction(import_act)

        # Separate input from output
        dp_menu.addSeparator()

        # Save DataPattern
        save_act = QtWidgets.QAction('&Save json', self)
        save_act.setStatusTip('Save as a DataPattern.json file')
        save_act.triggered.connect(self.dpcontroler.save_dp_call)
        dp_menu.addAction(save_act)

        # Export as ascii pattern matrix
        exportascii_act = QtWidgets.QAction('&Export ascii', self)
        exportascii_act.setStatusTip('Export as an ascii file')
        exportascii_act.triggered.connect(self.dpcontroler.exportascii_dp_call)
        dp_menu.addAction(exportascii_act)

        # Export as origin pattern matrix
        exportorigin_act = QtWidgets.QAction('&Export binary', self)
        exportorigin_act.setStatusTip('Export as a binary file')
        exportorigin_act.triggered.connect(self.dpcontroler.exportorigin_dp_call)
        dp_menu.addAction(exportorigin_act)

        # Save as image
        saveimage_act = QtWidgets.QAction('&Save Image', self)
        saveimage_act.setStatusTip('Save pattern as an image')
        saveimage_act.triggered.connect(self.dpcontroler.saveasimage_dp_call)
        dp_menu.addAction(saveimage_act)

        return dp_menu

    def untoggle_pb_maskrectangle(self):
        if self.pb_maskrectangle.isChecked():
            self.pb_maskrectangle.setChecked(False)
            self.dpcontroler.call_pb_maskrectangle(self.pb_maskrectangle)

    def untoggle_pb_maskpixel(self):
        if self.pb_maskpixel.isChecked():
            self.pb_maskpixel.setChecked(False)
            self.dpcontroler.call_pb_maskpixel(self.pb_maskpixel)

    def set_datapattern(self, datapattern):
        self.dpcontroler.set_datapattern(datapattern)

    def get_datapattern(self):
        return self.dpcontroler.get_datapattern()

    def are_changes_saved(self):
        return self.dpcontroler.are_changes_saved()


class DataPatternControler(QtCore.QObject):
    """ Data pattern controler class"""

    datapattern_opened = QtCore.pyqtSignal()
    datapattern_changed_or_saved = QtCore.pyqtSignal()

    def __init__(self, parent_widget=None, mpl_layout=None, infotext_box=None):

        super(DataPatternControler, self).__init__()

        self.parent_widget = parent_widget
        self.mainwindow = self.parent_widget.mainwindow
        self.infotext = infotext_box

        # Set config section
        if not config.parser.has_section('datapattern'):
            config.parser.add_section('datapattern')

        default_percentiles = [0.05, 0.99] if not config.parser.has_option('datapattern', 'default_percentiles') else \
            config.getlist('datapattern', 'default_percentiles')

        # initiate variables
        self.datapattern = None
        self.percentiles = default_percentiles
        self.ticks = None
        self.changes_saved = True

        # mpl variables
        self.mpl_layout = mpl_layout
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

        # Set up matplotlib canvas
        # get background color from color from widget and convert it to RBG
        pyqt_bkg = self.parent_widget.mainwindow.palette().color(QtGui.QPalette.Background).getRgbF()
        mpl_bkg = mpl.colors.rgb2hex(pyqt_bkg)

        # self.pltfig = plt.figure() # don't use pyplot
        #print(dir(mpl.figure))
        self.pltfig = mpl.figure.Figure()
        self.pltfig.set_facecolor(mpl_bkg)
        self.plot_ax = self.pltfig.add_subplot(111)
        self.plot_ax.set_aspect('equal')
        self.colorbar_ax = None
        self.addmpl(self.pltfig)
        # call tight_layout after addmpl
        self.pltfig.tight_layout()

    def addmpl(self, fig):
        if self.mpl_layout is None:
            return

        self.mpl_canvas = FigureCanvas(fig)
        self.mpl_layout.addWidget(self.mpl_canvas)
        self.mpl_canvas.draw()

        self.mpl_toolbar = NavigationToolbar(self.mpl_canvas,
                                             self.parent_widget, coordinates=False)
        self.mpl_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.mpl_layout.addWidget(self.mpl_toolbar)

        # connect status bar coordinates display
        self.mpl_canvas.mpl_connect('motion_notify_event', self.on_move)

    def refresh_mpl_color(self, new_mpl_bkg=None):
        # get background color from color from widget and convert it to RBG
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
        self.changes_saved = False
        self.datapattern_changed_or_saved.emit()

    def get_datapattern(self):
        if self.datapattern is not None:
            return self.datapattern.copy()
        else:
            return None

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
            self.datapattern_opened.emit()
            self.changes_saved = True
            self.datapattern_changed_or_saved.emit()

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
            self.datapattern_opened.emit()
            self.changes_saved = True
            self.datapattern_changed_or_saved.emit()

            # update config
            open_path = os.path.dirname(filename[0])
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
        self.changes_saved = True
        self.datapattern_changed_or_saved.emit()

        # update config
        save_path = os.path.dirname(filename[0])
        config.parser['datapattern']['save_path'] = save_path

    def import_dp_call(self):
        open_path = '' if not config.parser.has_option('datapattern', 'open_path') else \
            config.get('datapattern', 'open_path')
        filename = QtWidgets.QFileDialog.getOpenFileName(self.parent_widget,
                                                         'Import matrix file',
                                                         directory=open_path,
                                                         filter='Import matrix (*.txt *.csv *.2db)',
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename == ('', ''):  # Cancel
            return

        import_options = ('Single chip', 'Timepix quad')
        item, ok = QtWidgets.QInputDialog.getItem(self.parent_widget, "Select import format",
                                                  "Import format", import_options, 0, False)

        if not ok:
            return

        try:
            # TODO do orientations
            if item == 'Single chip':
                self.datapattern = pyfdd.DataPattern(file_path=filename[0], nChipsX=1, nChipsY=1, real_size=1)
            elif item == 'Timepix quad':
                self.datapattern = pyfdd.DataPattern(file_path=filename[0], nChipsX=2, nChipsY=2, real_size=3)
        except:
            QtWidgets.QMessageBox.warning(self.parent_widget, 'Warning message',
                                          'Error while importing the data.')
        else:
            # Draw pattern and update info text
            self.draw_new_datapattern()
            self.update_infotext()
            self.changes_saved = True
            self.datapattern_changed_or_saved.emit()

            # update config
            open_path = os.path.dirname(filename[0])
            config.parser['datapattern']['open_path'] = open_path

    def exportascii_dp_call(self):
        if not self.datapattern_exits():
            return

        save_path = '' if not config.parser.has_option('datapattern', 'save_path') else \
            config.get('datapattern', 'save_path')

        filename = QtWidgets.QFileDialog.getSaveFileName(self.parent_widget,
                                                         'Export DataPattern',
                                                         directory=save_path,
                                                         filter='ASCII (*.txt)',
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename == ('', ''):  # Cancel
            return

        self.datapattern.io_save_ascii(filename[0])

        # update config
        save_path = os.path.dirname(filename[0])
        config.parser['datapattern']['save_path'] = save_path

    def exportorigin_dp_call(self):
        if not self.datapattern_exits():
            return

        save_path = '' if not config.parser.has_option('datapattern', 'save_path') else \
            config.get('datapattern', 'save_path')

        filename = QtWidgets.QFileDialog.getSaveFileName(self.parent_widget,
                                                         'Export DataPattern',
                                                         directory=save_path,
                                                         filter='Binary (*.2db)',
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename == ('', ''):  # Cancel
            return

        self.datapattern.io_save_origin(filename[0])

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

    def are_changes_saved(self):
        return self.changes_saved

    def draw_new_datapattern(self):
        self.ticks = None
        self.draw_datapattern()

    def draw_datapattern(self):

        # Clear previous axes and colorbar
        if self.colorbar_ax is not None:
            self.colorbar_ax.remove()
        if self.plot_ax is not None:
            self.plot_ax.clear()

        if self.ticks is None:
            self.ticks = self.datapattern.get_ticks(self.percentiles)

        self.plot_ax, self.colorbar_ax = \
            self.datapattern.draw(self.plot_ax, ticks=self.ticks, **self.plot_labels)

        # call a few times to keep the figure from moving
        self.pltfig.tight_layout()
        self.pltfig.tight_layout()
        self.pltfig.tight_layout()
        self.pltfig.tight_layout()
        self.pltfig.tight_layout()
        self.mpl_canvas.draw()
        # self.plot_ax.set_aspect('equal')

    def update_infotext(self):
        if self.infotext is None:
            #raise warnings.warn('Info text box is not set')
            return

        base_text = 'Total counts: {:.1f}; Valid: {:.1f}\n' \
                    'Active pixels: {:d}; Masked: {:d}\n' \
                    'Angular range (x, y): {:.1f}, {:.1f}\n' \
                    'Pattern orientation (x, y, phi): {:.2f}, {:.2f}, {:.1f}'

        total_c = self.datapattern.matrixCurrent.data.sum()
        total_c_active = self.datapattern.matrixCurrent.sum()
        masked_p = self.datapattern.matrixCurrent.mask.sum()
        active_p = (~self.datapattern.matrixCurrent.mask).sum()
        xm = self.datapattern.xmesh[0, :]
        ym = self.datapattern.ymesh[:, 0]
        x_range = xm[-1] - xm[0]
        y_range = ym[-1] - ym[0]
        x_orient, y_orient = self.datapattern.center
        phi = self.datapattern.angle

        text = base_text.format(total_c, total_c_active, active_p, masked_p, x_range, y_range, x_orient, y_orient, phi)

        self.infotext.setText(text)

    def on_move(self, event):
        if event.inaxes == self.plot_ax:
            x, y = event.xdata, event.ydata
            if self.datapattern is not None:
                i, j = self.get_index_from_xy(x, y)
                # get value with 2 decimal cases. 2db files don't round properly
                z = self.datapattern.matrixDrawable[i, j]
                if isinstance(z, float):
                    z = float('{:.1f}'.format(z))
            else:
                z = 0

            self.mainwindow.statusBar().showMessage('(x,y,z) - ({:.2f},{:.2f},{})'.format(x, y, z))
        else:
            self.mainwindow.statusBar().showMessage('')

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
        #i = (np.abs(ym - y)).argmin()  # lines

        return i, j

    def on_maskpixelclick(self, event):
        if event.button == 1:
            x, y = event.xdata, event.ydata
            i, j = self.get_index_from_xy(x, y)
            self.datapattern.mask_pixel(i, j)

            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()
            self.changes_saved = False
            self.datapattern_changed_or_saved.emit()

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
        self.changes_saved = False
        self.datapattern_changed_or_saved.emit()

    def call_pb_maskrectangle(self, pushbutton):
        self.pb_maskrectangle = pushbutton
        if not self.datapattern_exits():
            self.pb_maskrectangle.setChecked(False)
            return

        if self.pb_maskrectangle.isChecked():
            rectprops = dict(facecolor='red', edgecolor='black',
                             alpha=0.8, fill=True)
            self.rselect_mpl = RectangleSelector(self.plot_ax, self.on_rectangleselect, drawtype='box', useblit=True,
                                                 interactive=False,
                                                 rectprops=rectprops)
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
            self.changes_saved = False
            self.datapattern_changed_or_saved.emit()

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
            self.changes_saved = False
            self.datapattern_changed_or_saved.emit()

            # update config
            config.parser['datapattern']['mask_above'] = str(value)

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

            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()
            self.changes_saved = False
            self.datapattern_changed_or_saved.emit()

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
            self.changes_saved = False
            self.datapattern_changed_or_saved.emit()

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
            self.changes_saved = False
            self.datapattern_changed_or_saved.emit()

            # update config
            config.parser['datapattern']['expand_mask'] = str(value)

    def call_pb_clearmask(self):
        if not self.datapattern_exits():
            return

        self.datapattern.clear_mask()

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()
        self.changes_saved = False
        self.datapattern_changed_or_saved.emit()

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
        self.changes_saved = False
        self.datapattern_changed_or_saved.emit()

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
        self.datapattern.save_mask(filename[0])

        # update config
        mask_path = os.path.dirname(filename[0])
        config.parser['datapattern']['save_path'] = mask_path

    def call_pb_buildmesh(self):
        if not self.datapattern_exits():
            return

        buildmesh_dialog = BuildMesh_dialog(parent_widget=self.parent_widget, dp_controler=self)

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
        self.changes_saved = False
        self.datapattern_changed_or_saved.emit()

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
            self.changes_saved = False
            self.datapattern_changed_or_saved.emit()

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
        self.changes_saved = False
        self.datapattern_changed_or_saved.emit()

    def call_pb_orientchanneling(self, pushbutton):
        self.pb_orientchanneling = pushbutton
        if not self.datapattern_exits():
            self.pb_orientchanneling.setChecked(False)
            return

        if self.pb_orientchanneling.isChecked():
            self.ang_wid = AngleMeasure(self.plot_ax, self.callonangle)
            self.use_crosscursor_in_axes(True)
        else:
            self.ang_wid = None
            self.use_crosscursor_in_axes(False)

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
            self.changes_saved = False
            self.datapattern_changed_or_saved.emit()

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
            self.changes_saved = False
            self.datapattern_changed_or_saved.emit()
        else:
            pass
            # print('Cancelled')


def main():
    app = QtWidgets.QApplication(sys.argv)
    # window = DataPattern_widget()
    window = DataPattern_window()
    window.show()
    print(window.size())
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
