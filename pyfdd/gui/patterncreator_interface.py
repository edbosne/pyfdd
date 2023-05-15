import sys
import os
import warnings
from enum import Enum, IntEnum
import json

from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from PySide2 import QtCore, QtGui, QtWidgets, uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT  # as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from pyfdd.core.datapattern.plot_widgets import AngleMeasurement
import matplotlib.pyplot as plt  # do not use pyplot
import matplotlib as mpl
import seaborn as sns
import numpy as np

import pyfdd
from pyfdd.core.datapattern.datapattern import create_detector_mesh

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from pyfdd.gui.qt_designer.patterncreator_widget import Ui_PatternCreatorWidget
from pyfdd.gui.qt_designer.creatorconfig_dialog import Ui_CreatorConfigDialog
from pyfdd.gui.fitmanager_interface import FitParameter, SiteRange, FitParameterDynamicLayout, SiteRangeDynamicLayout
from pyfdd.gui.datapattern_controler import DataPatternControler, BuildMesh_dialog
from pyfdd.gui.datapattern_interface import DataPattern_window

import pyfdd.gui.config as config


class GenMethod(IntEnum):
    channeling_yield = 0
    ideal = 1
    poisson = 2
    montecarlo = 3


class CreatorConfig_dialog(QtWidgets.QDialog, Ui_CreatorConfigDialog):

    def __init__(self, parent_widget, current_config):
        assert isinstance(current_config, dict)
        super(CreatorConfig_dialog, self).__init__(parent_widget)
        self.setupUi(self)

        self.load_config(current_config)
        self.new_config = dict()

        # set regular expression validator
        # accepts floats or integers ex.: 1, 1.1, 1000
        reg_ex = QtCore.QRegExp(
            r'^\d+(\.\d+)?$')

        input_validator = QtGui.QRegExpValidator(reg_ex, parent=self.le_normalization)
        self.le_normalization.setValidator(input_validator)

        # Set up le normalization
        self.call_enable_normalization()
        self.call_normalization_min()

        # Connect signals
        self.cb_gen_method.currentIndexChanged.connect(self.call_enable_normalization)
        self.cb_gen_method.currentIndexChanged.connect(self.call_normalization_min)

    def load_config(self, config):
        self.sb_numsites.setValue(config['n_sites'])
        self.sb_subpixels.setValue(config['sub_pixels'])
        self.le_normalization.setText(str(config['normalization']))
        self.cb_gen_method.setCurrentIndex(config['gen_method'])

    def get_config(self):
        n_sites = self.sb_numsites.value()
        sub_pixels = self.sb_subpixels.value()
        gen_method = GenMethod(self.cb_gen_method.currentIndex())
        # Don't set up a normalization if what you are looking for is channeling yield.
        normalization = float(self.le_normalization.text()) if not gen_method == 0 else 0

        self.new_config = {
            'n_sites': n_sites,
            'sub_pixels': sub_pixels,
            'normalization': normalization,
            'gen_method': gen_method}

        return self.new_config

    def call_enable_normalization(self):

        if self.cb_gen_method.currentText() == 'Channeling yield':
            self.le_normalization.setEnabled(False)
        else:
            self.le_normalization.setEnabled(True)

    def call_normalization_min(self):

        if not self.cb_gen_method.currentText() == 'Channeling yield':
            normalization = float(self.le_normalization.text())
            if normalization <= 0:
                self.le_normalization.setText('1')


class CreatorParametter(FitParameter):

    def __init__(self, parent_widget, key='', name='par', initial_value=0,
                 pb_edit=None, lb_name=None, lb_description=None, **kwargs):

        for kw in kwargs.keys():
            if kw in ('bounds', 'step_modifier', 'fixed'):
                # Ignore
                pass
            else:
                raise ValueError('Unexpected keyword argument, {}'.format(kw))

        super(CreatorParametter, self).__init__(parent_widget, key=key, name=name, initial_value=initial_value,
                                                pb_edit=pb_edit, lb_name=lb_name, lb_description=lb_description)
        self.parent = parent_widget
        self.update_description()

    def update_description(self):
        if self.initial_value < 100:
            base_text = '{:.2f}'
        else:
            base_text = '{:.1e}'
        text = base_text.format(self.initial_value)
        self.lb_description.setText(text)

    def call_pb_edit(self):
        if self.name == 'sigma' or self.name.startswith('fraction'):
            minimum = 0
        else:
            minimum = -9999

        new_initial_value, ok = QtWidgets.QInputDialog.getDouble(self.parent, 'Parameter Value',
                                                                 f'Insert the value for {self.name}\t\t\t',
                                                                 value=self.initial_value,
                                                                 min=minimum,
                                                                 decimals=2)
        if ok:
            self.initial_value = new_initial_value
            self.update_description()
        else:
            # Canceled
            pass


class PatternCreator_window(QtWidgets.QMainWindow):
    """ Class to use the data pattern widget in a separate window"""

    def __init__(self, *args, **kwargs):
        super(PatternCreator_window, self).__init__(*args, **kwargs)

        # Load configuration
        if config.parser is None:
            config.filename = 'patterncreator_config.ini'
            config.read()

        # Setup the window
        self.setWindowTitle("Pattern Creator")
        self.statusBar()

        # Set a Pattern Creator widget as central widget
        dp_w = PatternCreator_widget(mainwindow=self)
        self.setCentralWidget(dp_w)
        self.resize(1150, 670)

    @staticmethod
    def get_datapattern():
        datapattern = pyfdd.DataPattern(
            '../../test_pyfdd/data_files/pad_dp_2M.json')
        return datapattern

    @staticmethod
    def get_simlibrary():
        simlibrary = pyfdd.Lib2dl(
            '../../test_pyfdd/data_files/sb600g05.2dl')
        return simlibrary

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        config.write()


class PatternCreator_widget(QtWidgets.QWidget, Ui_PatternCreatorWidget):
    """ Data pattern widget class"""

    simlibrary_opened = QtCore.pyqtSignal()

    def __init__(self, *args, mainwindow=None, **kwargs):
        """
        Init method for the data pattern widget
        :param args:
        :param mainwindow: Main window object
        :param kwargs:
        """

        super(PatternCreator_widget, self).__init__(*args, **kwargs)

        # Alternative way to load the ui created with PyQt creator
        # uic.loadUi('qt_designer/datapattern_widget.ui', self)

        self.setupUi(self)
        self.mainwindow = mainwindow

        # Set config section
        if not config.parser.has_section('patterncreator'):
            config.parser.add_section('patterncreator')

        # set the mpl widget background colour
        self.mplwindow.setStyleSheet('background: palette(window);')

        # Instantiate datapattern controler
        self.dpcontroler = DataPatternControler(parent_widget=self)#, mpl_layout=self.mplvl, infotext_box=None)
        self.mainwindow = mainwindow
        self.mpl_layout = self.mplvl
        self.dpcontroler.set_widgets_and_layouts(mpl_layout=self.mplvl, mainwindow=self.mainwindow)

        # Create a menubar entry for the datapattern
        self.menubar = self.mainwindow.menuBar()
        self.dp_menu = self.setup_menu()

        # Popup widgets that need a reference in self
        self.dp_external = []

        # Variables
        self.tr_gen_method = {'channeling_yield': 'Channeling yield',
                              'ideal': 'Ideal',
                              'poisson': 'Poisson noise',
                              'montecarlo': 'Monte Carlo'}

        self.simlibrary = None
        self.get_simlibrary()
        self.changes_saved = True
        self.pattern_mesh_isset = None
        self.pattern_xmesh = None
        self.pattern_ymesh = None

        # Fit configuration
        default_creatorconfig = {'n_sites': 1,
                                 'sub_pixels': 1,
                                 'normalization': 1,
                                 'gen_method': GenMethod.channeling_yield}

        if not config.parser.has_option('patterncreator', 'creatorconfig'):
            self.creatorconfig = default_creatorconfig.copy()
        else:
            self.creatorconfig = config.getdict('patterncreator', 'creatorconfig')
            # convert ints to enum
            self.creatorconfig['gen_method'] = GenMethod(self.creatorconfig['gen_method'])

        # Sites ranges
        self.dynamic_site_ranges = SiteRangeDynamicLayout(parent_widget=self,
                                                          grid_layout=self.sitesrange_layout,
                                                          lb_f1_name=self.lb_f1_name,
                                                          le_site1=self.le_site1,
                                                          multiple_sites=False)

        # Parameters
        # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        self.dynamic_parameters = FitParameterDynamicLayout(parent_widget=self,
                                                            grid_layout=self.parameters_layout,
                                                            exclude=('reset', 'total_cts'),
                                                            fitparameter_type=CreatorParametter)

        # Connect signals
        self.pb_creatorconfig.clicked.connect(self.call_pb_creatorconfig)
        self.pb_buildmesh.clicked.connect(self.call_pb_buildmesh)
        self.pb_get_mesh_from_dp.clicked.connect(self.call_pb_get_mesh_from_dp)
        self.pb_get_mesh_from_bkg.clicked.connect(self.call_pb_get_mesh_from_bkg)
        self.pb_generatepattern.clicked.connect(self.call_pb_generatepattern)
        self.pb_opendatapattern.clicked.connect(self.call_pb_opendatapattern)

        self.update_infotext()
        self.update_n_sites_widgets()

        # Pattern visualization
        self.pb_colorscale.clicked.connect(self.dpcontroler.call_pb_colorscale)
        self.pb_setlabels.clicked.connect(self.dpcontroler.call_pb_setlabels)

    def setup_menu(self):
        dp_menu = self.menubar.addMenu('&Patt. Creator')

        # Export pattern matrix
        export_act = QtWidgets.QAction('&Export pattern', self)
        export_act.setStatusTip('Export as an .txt .csv or .2db file')
        export_act.triggered.connect(self.dpcontroler.export_dp_call)
        dp_menu.addAction(export_act)

        # Save as image
        saveimage_act = QtWidgets.QAction('&Save Image', self)
        saveimage_act.setStatusTip('Save pattern as an image')
        saveimage_act.triggered.connect(self.dpcontroler.saveasimage_dp_call)
        dp_menu.addAction(saveimage_act)

        # Copy to clipboard
        copy_act = QtWidgets.QAction('&Copy to clipboard', self)
        copy_act.setStatusTip('Copy simulation image to clipboard')
        copy_act.triggered.connect(self.dpcontroler.call_copy_to_clipboard)
        dp_menu.addAction(copy_act)

        return dp_menu

    def update_infotext(self):

        base_text = 'Pattern mesh set: {}; Library set: {}\n' \
                    'Number of sites: {}; Sub-pixels: {}\n' \
                    'Normalization: {}\n' \
                    'Generator method: {}'

        mesh_set = False if self.pattern_mesh_isset is None else True
        lib_set = False if self.simlibrary is None else True
        n_sites = self.creatorconfig['n_sites']
        sub_pixels = self.creatorconfig['sub_pixels']
        normalization = self.creatorconfig['normalization']
        normalization = normalization if not normalization == 0 else '---'  # Change normalization 0 to '---'
        gen_method_code = self.creatorconfig['gen_method'].name
        gen_method = self.tr_gen_method[gen_method_code]

        text = base_text.format(mesh_set,
                                lib_set,
                                n_sites,
                                sub_pixels,
                                normalization,
                                gen_method)
        self.infotext.setText(text)

    def get_datapattern(self):

        if self.mainwindow is None:
            datapattern = None
        else:
            datapattern = self.mainwindow.get_datapattern()
        return datapattern

    def get_backgroundpattern(self):

        if self.mainwindow is None:
            datapattern = None
        else:
            datapattern = self.mainwindow.get_bkgtab_datapattern()
        return datapattern

    def get_background_pattern_and_factor(self):
        if self.mainwindow is None:
            return None, None
        elif hasattr(self.mainwindow, 'get_background_pattern_and_factor'):
            return self.mainwindow.get_background_pattern_and_factor()
        else:
            return None, None

    def get_simlibrary(self):

        if self.mainwindow is None:
            self.simlibrary = None
        else:
            self.simlibrary = self.mainwindow.get_simlibrary()

    def are_changes_saved(self):
        return self.changes_saved

    def set_creatorconfig_values(self, n_sites=None, sub_pixels=None, normalization=None, gen_method=None):
        if n_sites is not  None:
            self.creatorconfig['n_sites'] = n_sites
        if sub_pixels is not None:
            self.creatorconfig['sub_pixels'] = sub_pixels
        if normalization is not None:
            self.creatorconfig['normalization'] = normalization
        if gen_method is not None:
            self.creatorconfig['gen_method'] = gen_method

        self.update_infotext()
        self.update_n_sites_widgets()
        config.parser['patterncreator']['creatorconfig'] = json.dumps(self.creatorconfig)

    def set_sites(self, sites_list):
        self.dynamic_site_ranges.set_sites(sites_list)

    def set_parameters(self, parameters_dict):
        self.dynamic_parameters.update_initial_values(parameters_dict=parameters_dict)

    def call_pb_creatorconfig(self):

        creatorconfig_dialog = CreatorConfig_dialog(parent_widget=self, current_config=self.creatorconfig)
        if creatorconfig_dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.creatorconfig = creatorconfig_dialog.get_config()
            self.update_infotext()
            self.update_n_sites_widgets()
            config.parser['patterncreator']['creatorconfig'] = json.dumps(self.creatorconfig)
        else:
            # Canceled
            pass

    def call_pb_buildmesh(self):

        buildmesh_dialog = BuildMesh_dialog(parent_widget=self, dp_controler=None,
                                            config_section='patterncreator', npixels_active=True)

        if buildmesh_dialog.exec_() == QtWidgets.QDialog.Accepted:
            output = buildmesh_dialog.get_settings()
            n_h = output['horizontal pixels']
            n_v = output['vertical pixels']
            if output['selected'] == 'detector':
                pixel_size = output['pixel size']
                distance = output['distance']
                self.pattern_xmesh, self.pattern_ymesh = \
                    create_detector_mesh(n_h_pixels=n_h, n_v_pixels=n_v,
                                         pixel_size=pixel_size, distance=distance)
                self.pattern_mesh_isset = True

            elif output['selected'] == 'step':
                angular_step = output['angular step']
                self.pattern_xmesh, self.pattern_ymesh = \
                    create_detector_mesh(n_h_pixels=n_h, n_v_pixels=n_v,
                                         d_theta=angular_step)
                self.pattern_mesh_isset = True
            else:
                warnings.warn('Non valid selection')
        else:
            pass
            # print('Cancelled')

        buildmesh_dialog.deleteLater()

        # Update info text
        self.update_infotext()

    def call_pb_get_mesh_from_dp(self):

        datapattern = self.get_datapattern()

        if datapattern is None:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'The DataPattern does not exist.')
            return

        self.set_mesh_from_dp(datapattern)

    def call_pb_get_mesh_from_bkg(self):
        datapattern = self.get_backgroundpattern()

        if datapattern is None:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'The Background DataPattern does not exist.')
            return

        self.set_mesh_from_dp(datapattern)

    def set_mesh_from_dp(self, datapattern):

        assert isinstance(datapattern, pyfdd.DataPattern)

        self.pattern_xmesh, self.pattern_ymesh = \
            datapattern.get_xymesh()
        self.pattern_mesh_isset = True

        # update config

        horizontal_pixels = datapattern.nx
        vertical_pixel = datapattern.ny
        angular_step = (self.pattern_xmesh[0, -1] - self.pattern_xmesh[0, 0]) / (horizontal_pixels - 1)
        angular_step = np.round(angular_step, decimals=2)

        default_mesh_settings = {'horizontal pixels': 256,
                                 'vertical pixels': 256,
                                 'pixel size': 0.055,
                                 'distance': 315.0,
                                 'angular step': 0.1,
                                 'selected': 'detector'}
        mesh_setting = config.getdict('patterncreator', 'mesh_settings') if \
            config.parser.has_option('patterncreator', 'mesh_settings') else \
            default_mesh_settings
        mesh_setting['horizontal pixels'] = horizontal_pixels
        mesh_setting['vertical pixels'] = vertical_pixel
        mesh_setting['angular step'] = angular_step
        mesh_setting['selected'] = 'step'
        config.parser['patterncreator']['mesh_settings'] = json.dumps(mesh_setting)

        # Update info text
        self.update_infotext()

    def call_pb_generatepattern(self):

        # Initial checks
        if not self.pattern_mesh_isset:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'The Pattern mesh is not set.')
            return

        if self.simlibrary is None:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'The Simulations Library is not set.')
            return

        background_pattern, factor = self.get_background_pattern_and_factor()

        if background_pattern is not None:
            try:
                pyfdd.BackgroundTools.verify_mesh(background_array=background_pattern,
                                                  xmesh=self.pattern_xmesh, ymesh=self.pattern_ymesh)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Warning message', str(e))
                return


        # Get values
        n_sites = self.creatorconfig['n_sites']
        sub_pixels = self.creatorconfig['sub_pixels']
        normalization = self.creatorconfig['normalization']
        gen_method = self.creatorconfig['gen_method'].name
        gen_method = 'yield' if gen_method == 'channeling_yield' else gen_method

        sites = self.dynamic_site_ranges.get_sites_for_fit()
        sites = [s[0] for s in sites]  # remove sublist
        initial_values = self.dynamic_parameters.get_initial_values()

        dx = initial_values['dx']
        dy = initial_values['dy']
        phi = initial_values['phi']
        fractions_per_sim = list()
        for fn in range(len(sites)):
            fraction_number = fn + 1
            fraction_name = 'f_p{}'.format(fraction_number)
            fractions_per_sim.append(initial_values[fraction_name])

        sigma = initial_values['sigma']

        gen = pyfdd.PatternCreator(self.simlibrary,
                                   self.pattern_xmesh,
                                   self.pattern_ymesh,
                                   simulations=sites,
                                   sub_pixels=sub_pixels,
                                   mask_out_of_range=True,
                                   background_pattern=background_pattern,
                                   background_factor=factor)

        pattern = gen.make_pattern(dx, dy, phi, fractions_per_sim,
                                   total_events=normalization,
                                   sigma=sigma,
                                   pattern_type=gen_method)

        generated_pattern = pyfdd.DataPattern(pattern_array=pattern)
        generated_pattern.set_xymesh(self.pattern_xmesh, self.pattern_ymesh)

        self.dpcontroler.set_datapattern(generated_pattern)

    def call_pb_opendatapattern(self):

        datapattern = self.dpcontroler.get_datapattern()

        if datapattern is not None:
            new_dp_window = DataPattern_window()
            new_dp_window.set_datapattern(datapattern)
            new_dp_window.setWindowTitle('Generated Pattern')
            new_dp_window.show()
            self.dp_external.append(new_dp_window)

        else:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'A pattern has not been generated.')

    def update_all(self):
        self.get_datapattern()
        self.get_simlibrary()
        self.update_infotext()

    def update_n_sites_widgets(self):
        self.dynamic_site_ranges.update_n_sites_widgets(self.creatorconfig['n_sites'])
        self.dynamic_parameters.update_n_sites_widgets(self.creatorconfig['n_sites'])


def main():
    app = QtWidgets.QApplication(sys.argv)
    # window = DataPattern_widget()
    window = PatternCreator_window()
    window.show()
    print(window.size())
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
