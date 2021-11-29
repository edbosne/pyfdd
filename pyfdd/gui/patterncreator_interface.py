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

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from pyfdd.gui.qt_designer.patterncreator_widget import Ui_PatternCreatorWidget
from pyfdd.gui.qt_designer.creatorconfig_dialog import Ui_CreatorConfigDialog
from pyfdd.gui.fitmanager_interface import FitParameter, SiteRange, FitParameterDynamicLayout, SiteRangeDynamicLayout
from pyfdd.gui.datapattern_interface import DataPatternControler
import pyfdd.gui.config as config


class GenMethod(IntEnum):
    channeling_yield = 0
    ideal = 1
    poisson = 2
    monte_carlo = 3


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

    def load_config(self, config):
        self.sb_numsites.setValue(config['n_sites'])
        self.sb_subpixels.setValue(config['sub_pixels'])
        self.le_normalization.setText(str(config['normalization']))
        self.cb_gen_method.setCurrentIndex(config['gen_method'])

    def get_config(self):
        self.new_config = {
            'n_sites': self.sb_numsites.value(),
            'sub_pixels': self.sb_subpixels.value(),
            'normalization': float(self.le_normalization.text()),
            'gen_method': GenMethod(self.cb_gen_method.currentIndex())}

        return self.new_config


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
        new_initial_value, ok = QtWidgets.QInputDialog.getDouble(self.parent, 'Parameter Value',
                                                                 f'Insert the value for {self.name}\t\t\t',
                                                                 value=self.initial_value, decimals=2)
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
        self.dpcontroler = DataPatternControler(parent_widget=self, mpl_layout=self.mplvl, infotext_box=None)

        # Create a menubar entry for the datapattern
        self.menubar = self.mainwindow.menuBar()
        self.dp_menu = self.setup_menu()

        # Popup widgets that need a reference in self
        self.dp_external = []

        # Variables
        self.tr_gen_method = {'channeling_yield': 'Channeling yield',
                              'ideal': 'Ideal',
                              'poisson': 'Poisson noise',
                              'monte_carlo': 'Monte Carlo'}

        self.simlibrary = None
        self.get_simlibrary()
        self.changes_saved = True
        self.pattern_mesh = None

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
        # self.pb_reset.clicked.connect(lambda: self.refresh_parameters(reset=True))
        # self.pb_abortfits.setEnabled(False)
        # self.pb_runfits.clicked.connect(self.call_pb_runfits)
        # self.pb_abortfits.clicked.connect(self.call_pb_abortfits)
        # self.pb_viewresults.clicked.connect(self.call_pb_viewresults)
        # self.pb_savetable.clicked.connect(self.call_pb_savetable)
        # self.pb_viewfit.clicked.connect(self.call_pb_viewlastfit)
        # self.pb_viewfitdiff.clicked.connect(self.call_pb_viewfitdiff)
        # self.pb_filldata.clicked.connect(self.call_pb_filldata)

        self.update_infotext()
        self.update_n_sites_widgets()

        # Pattern visualization
        self.pb_colorscale.clicked.connect(self.dpcontroler.call_pb_colorscale)
        self.pb_setlabels.clicked.connect(self.dpcontroler.call_pb_setlabels)

    def setup_menu(self):
        dp_menu = self.menubar.addMenu('&Patt. Creator')

        # Export as ascii pattern matrix
        exportascii_act = QtWidgets.QAction('&Export ascii', self)
        exportascii_act.setStatusTip('Export as an ascii file')
        exportascii_act.triggered.connect(self.dpcontroler.exportascii_dp_call)
        dp_menu.addAction(exportascii_act)

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

    def init_parameters(self):
        # Parameters
        # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        # dx
        par = CreatorParametter(parent_widget=self, key='dx', name='dx',
                                pb_edit=self.pb_dx, lb_name=self.lb_dx_name,
                                lb_description=self.lb_dx)
        self.parameter_objects.append(par)
        # dy
        par = CreatorParametter(parent_widget=self, key='dy', name='dy',
                                pb_edit=self.pb_dy, lb_name=self.lb_dy_name,
                                lb_description=self.lb_dy)
        self.parameter_objects.append(par)
        # phi
        par = CreatorParametter(parent_widget=self, key='phi', name='phi',
                                pb_edit=self.pb_phi, lb_name=self.lb_phi_name,
                                lb_description=self.lb_phi)
        self.parameter_objects.append(par)
        # sigma
        par = CreatorParametter(parent_widget=self, key='sigma', name='sigma',
                                pb_edit=self.pb_sigma, lb_name=self.lb_sigma_name,
                                lb_description=self.lb_sigma)
        self.parameter_objects.append(par)
        # f_p1
        par = CreatorParametter(parent_widget=self, key='f_p1', name='fraction #1',
                                pb_edit=self.pb_f1, lb_name=self.lb_f1_name,
                                lb_description=self.lb_f1)
        self.parameter_objects.append(par)

    def refresh_parameters(self, reset: bool = False):
        """
        Refresh the parameters acoording to the current data pattern and library.
        :param reset: If true all paremeters that were changed by the user are reset.
        :return:
        """

        # Compute values
        fitparameters = pyfdd.FitParameters(n_sites=self.creatorconfig['n_sites'])

        parameter_keys = fitparameters.get_keys()
        parameter_keys.remove('total_cts')
        initial_values = fitparameters.get_initial_values()
        fixed_values = fitparameters.get_fixed_values()
        bounds = fitparameters.get_bounds()
        step_modifier = fitparameters.get_step_modifier()

        # Apply new values to the interface
        for key, parameter in zip(parameter_keys, self.parameter_objects):
            assert parameter.key == key
            if not reset and parameter.was_changed:
                # Keep the value introduced by the user
                continue
            else:
                parameter.reset_values_to(initial_value=initial_values[key],
                                          bounds=bounds[key],
                                          step_modifier=step_modifier[key],
                                          fixed=fixed_values[key])

    def init_sites_ranges(self):
        """ Create the first site range widget"""
        srange = SiteRange(parent_widget=self, key='sr1', name='Site #1', multiple_sites=False,
                           lb_name=self.lb_f1_name,
                           le_siterange=self.le_site1)
        self.sites_range_objects.append(srange)

    def update_infotext(self):

        base_text = 'Pattern mesh set: {}; Library set: {}\n' \
                    'Number of sites: {}; Sub-pixels: {}\n' \
                    'Normalization: {}\n' \
                    'Generator method: {}'

        mesh_set = False if self.pattern_mesh is None else True
        lib_set = False if self.simlibrary is None else True
        n_sites = self.creatorconfig['n_sites']
        sub_pixels = self.creatorconfig['sub_pixels']
        normalization = self.creatorconfig['normalization']
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
            self.datapattern = None
        else:
            self.datapattern = self.mainwindow.get_datapattern()

    def get_simlibrary(self):

        if self.mainwindow is None:
            self.simlibrary = None
        else:
            self.simlibrary = self.mainwindow.get_simlibrary()

    def are_changes_saved(self):
        return self.changes_saved

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

    def update_all(self):
        self.get_datapattern()
        self.get_simlibrary()
        self.update_infotext()
        self.refresh_parameters()

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
