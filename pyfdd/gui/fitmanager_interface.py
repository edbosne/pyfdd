
import sys
import os
import warnings
from enum import Enum, IntEnum

from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from PySide2 import QtCore, QtGui, QtWidgets, uic

import pyfdd

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from pyfdd.gui.qt_designer.fitmanager_widget import Ui_FitManagerWidget
from pyfdd.gui.qt_designer.fitconfig_dialog import Ui_FitConfigDialog
from pyfdd.gui.qt_designer.parameteredit_dialog import Ui_ParameterEditDialog


class Profile(IntEnum):
    coarse = 0
    default = 1
    fine = 2
    custom = 3


class CostFunc(IntEnum):
    chi2 = 0
    ml = 1


class FitConfig_dialog(QtWidgets.QDialog, Ui_FitConfigDialog):
    def __init__(self, parent_widget, current_config, fitman):
        assert isinstance(current_config, dict)
        assert isinstance(fitman, pyfdd.FitManager)
        super(FitConfig_dialog, self).__init__(parent_widget)
        self.setupUi(self)

        self.fitman = fitman
        self.load_config(current_config)
        self.new_config = dict

        # Connect signals
        self.cb_costfunc.currentIndexChanged.connect(self.update_profile_text)
        self.cb_profile.currentIndexChanged.connect(self.update_profile_text)

    def load_config(self, config):
        self.cb_costfunc.setCurrentIndex(config['cost_func'])
        self.ckb_geterrors.setChecked(config['get_errors'])
        self.sb_numsites.setValue(config['n_sites'])
        self.sb_subpixels.setValue(config['sub_pixels'])
        self.cb_profile.setCurrentIndex(int(config['min_profile']))
        if config['min_profile'] is Profile.custom:
            self.le_profile.setText(config['custom_profile'])
        else:
            self.update_profile_text()

    def update_profile_text(self):
        profile = Profile(self.cb_profile.currentIndex())
        cost_func = CostFunc(self.cb_costfunc.currentIndex())
        if profile is not Profile.custom:
            fit_options = self.fitman.profiles_fit_options[profile.name][cost_func.name]
            self.le_profile.setText(str(fit_options))
            self.le_profile.setReadOnly(True)
        else:
            self.le_profile.setReadOnly(False)

    def get_config(self):
        self.new_config = {
            'cost_func':CostFunc(self.cb_costfunc.currentIndex()),
            'get_errors':self.ckb_geterrors.isChecked(),
            'n_sites':self.sb_numsites.value(),
            'sub_pixels':self.sb_subpixels.value(),
            'min_profile':Profile(self.cb_profile.currentIndex())}

        if self.new_config['min_profile'] is Profile.custom:
            self.new_config['custom'] = self.le_profile.text()
        else:
            self.new_config['custom'] = ''

        return self.new_config


class FitManager_window(QtWidgets.QMainWindow):
    """ Class to use the data pattern widget in a separate window"""
    def __init__(self, *args, **kwargs):
        super(FitManager_window, self).__init__(*args, **kwargs)

        # Setup the window
        self.setWindowTitle("Fit Manager")
        self.statusBar()

        # Set a DataPattern widget as central widget
        dp_w = FitManager_widget(mainwindow=self)
        self.setCentralWidget(dp_w)
        self.resize(1150, 670)


class FitManager_widget(QtWidgets.QWidget, Ui_FitManagerWidget):
    """ Data pattern widget class"""

    def __init__(self, *args, mainwindow=None, **kwargs):
        """
        Init method for the data pattern widget
        :param args:
        :param mainwindow: Main window object
        :param kwargs:
        """

        super(FitManager_widget, self).__init__(*args, **kwargs)

        # Alternative way to load the ui created with PyQt creator
        # uic.loadUi('qt_designer/datapattern_widget.ui', self)

        self.setupUi(self)
        self.mainwindow = mainwindow

        # Create a menubar entry for the datapattern
        self.menubar = self.mainwindow.menuBar()
        self.dp_menu = self.setup_menu()

        # Variables
        self.tr_costfunc = {'chi2': 'Chi-square',
                            'ml': 'Neg. log likelihood'}
        self.datapattern = None
        self.simlibrary = None
        self.get_datapattern()
        self.get_simlibrary()
        self.fitman = None

        # Fit config
        self.fitconfig = {'cost_func': CostFunc.chi2,
                          'get_errors': True,
                          'n_sites': 1,
                          'sub_pixels': 1,
                          'min_profile': Profile.default,
                          'custom_profile': ''}
        self.update_fitman()

        # Parameters
        # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        self.parameter_keys = ()
        self.initial_values = dict()
        self.bounds = dict()
        self.step_modifier = dict()
        self.fixed = dict()
        self.sites_range = ()

        # Connect signals
        self.pb_fitconfig.clicked.connect(self.call_pb_fitconfig)

        self.update_infotext()

    def setup_menu(self):
        # nothing to do here at the moment
        return None

    def update_infotext(self):

        base_text = 'Data pattern set: {}; Library set: {}\n' \
                    'Cost function: {}; Get errors: {}\n' \
                    'Number of sites: {}; Sub-pixels: {}\n' \
                    'Miniminazion profile: {}'

        dp_set = False if self.datapattern is None else True
        lib_set = False if self.simlibrary is None else True
        cost_func = self.tr_costfunc[self.fitconfig['cost_func'].name]
        get_errors = self.fitconfig['get_errors']
        n_sites = self.fitconfig['n_sites']
        sub_pixels = self.fitconfig['sub_pixels']
        min_profile = self.fitconfig['min_profile'].name

        text = base_text.format(dp_set, lib_set, cost_func,
                                get_errors, n_sites,
                                sub_pixels,min_profile)

        self.infotext.setText(text)

    def get_datapattern(self):

        if self.mainwindow is None:
            self.datapattern = None
        else:
            # TODO
            pass

    def get_simlibrary(self):

        if self.mainwindow is None:
            self.simlibrary = None
        else:
            # TODO
            pass

    def call_pb_fitconfig(self):

        fitconfig_dialog = FitConfig_dialog(parent_widget=self, current_config=self.fitconfig, fitman=self.fitman)
        if fitconfig_dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.fitconfig = fitconfig_dialog.get_config()
            self.update_infotext()
            self.update_fitman()
        else:
            # Canceled
            pass

    def update_fitman(self):
        cost_function = self.fitconfig['cost_func'].name
        n_sites = self.fitconfig['n_sites']
        sub_pixels = self.fitconfig['sub_pixels']
        self.fitman = pyfdd.FitManager(cost_function=cost_function,
                                       n_sites=n_sites,
                                       sub_pixels=sub_pixels)


def main():
    app = QtWidgets.QApplication(sys.argv)
    # window = DataPattern_widget()
    window = FitManager_window()
    window.show()
    print(window.size())
    sys.exit(app.exec())


if __name__ == '__main__':
    main()