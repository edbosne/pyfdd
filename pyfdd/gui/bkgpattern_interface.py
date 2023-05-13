import sys
import os
import warnings
from enum import Enum, IntEnum
import numpy as np
import json

from PyQt5 import QtCore, QtGui, QtWidgets, uic, sip
# from PySide2 import QtCore, QtGui, QtWidgets, uic

import pyfdd

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from pyfdd.gui.qt_designer.bkgtools_widget import Ui_BkgToolsWidget
from pyfdd.gui.qt_designer.corrfactor_dialog import Ui_CorrFactorDialog
from pyfdd.gui.datapattern_interface import DataPattern_window, DataPatternControler

import pyfdd.gui.config as config


class CorrFactor_dialog(QtWidgets.QDialog, Ui_CorrFactorDialog):
    def __init__(self, parent_widget, corr_factor, bkg_counts=None):
        super(CorrFactor_dialog, self).__init__(parent_widget)
        self.setupUi(self)
        if bkg_counts is not None:
            bkg_counts = int(bkg_counts)

        self.corr_factor = corr_factor
        self.bkg_counts = bkg_counts if bkg_counts is not None else 0 if not config.parser.has_option('bkgpattern', 'bkg_counts') else \
            config.getlist('bkgpattern', 'bkg_counts')
        self.bkg_time = 3600 if not config.parser.has_option('bkgpattern','bkg_time') else \
            config.getlist('bkgpattern', 'bkg_time')
        self.data_counts = 1000000 if not config.parser.has_option('bkgpattern','data_counts') else \
            config.getlist('bkgpattern', 'data_counts')
        self.data_time = 3600 if not config.parser.has_option('bkgpattern', 'data_time') else \
            config.getlist('bkgpattern', 'data_time')

        self.le_data_time.setText(str(self.data_time))
        self.le_data_counts.setText(str(self.data_counts))
        self.le_bkg_time.setText(str(self.bkg_time))
        self.le_bkg_counts.setText(str(self.bkg_counts))
        self.le_factor.setText(str(self.corr_factor))

        validator0 = QtGui.QDoubleValidator(bottom=0)
        validator0.setLocale(QtCore.QLocale("en_US"))
        validator1 = QtGui.QDoubleValidator(bottom=1)
        validator1.setLocale(QtCore.QLocale("en_US"))
        self.le_data_time.setValidator(validator0)
        self.le_data_counts.setValidator(validator0)
        self.le_bkg_time.setValidator(validator0)
        self.le_bkg_counts.setValidator(validator0)
        self.le_factor.setValidator(validator1)

        self.pb_calculate.clicked.connect(self.call_pb_calculate)
        self.buttonBox.clicked.connect(self.closeEvent)

    def call_pb_calculate(self):
        self.data_time = float(self.le_data_time.text())
        self.data_counts = float(self.le_data_counts.text())
        self.bkg_time = float(self.le_bkg_time.text())
        self.bkg_counts = float(self.le_bkg_counts.text())

        try:
            factor = pyfdd.BackgroundTools.calculate_factor(data_time=self.data_time, data_cts=self.data_counts,
                                                        bkg_time=self.bkg_time, bkg_cts=self.bkg_counts)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Warning message', str(e))
            return

        self.corr_factor = factor
        self.le_factor.setText(str(self.corr_factor))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.data_time = float(self.le_data_time.text())
        self.data_counts = float(self.le_data_counts.text())
        self.bkg_time = float(self.le_bkg_time.text())
        self.bkg_counts = float(self.le_bkg_counts.text())
        self.corr_factor = float(self.le_factor.text())

        config.parser['bkgpattern']['data_time'] = str(self.data_time)
        config.parser['bkgpattern']['data_counts'] = str(self.data_counts)
        config.parser['bkgpattern']['bkg_time'] = str(self.bkg_time)
        config.parser['bkgpattern']['bkg_counts'] = str(self.bkg_counts)

        if isinstance(event, QtGui.QCloseEvent):
            event.accept()


class BkgTools_groupbox(QtWidgets.QWidget, Ui_BkgToolsWidget):

    def __init__(self, parent_widget):
        super(BkgTools_groupbox, self).__init__(parent_widget)
        self.setupUi(self)


class BkgPattern_window(QtWidgets.QMainWindow):
    """ Class to use the data pattern widget in a separate window"""
    def __init__(self, *args, **kwargs):
        super(BkgPattern_window, self).__init__(*args, **kwargs)

        # Load configuration
        if config.parser is None:
            config.filename = 'bkgpattern_config.ini'
            config.read()

        # Set up the window
        self.window_title = "Background Pattern"
        self.setWindowTitle(self.window_title)
        self.statusBar()

        # Set a BkgPattern widget as central widget
        self.bp_w = BkgPattern_widget(mainwindow=self)
        self.setCentralWidget(self.bp_w)
        self.resize(1150, 670)

        # Connect signals
        self.bp_w.datapattern_changed.connect(self.title_update)
        self.bp_w.datapattern_saved.connect(self.title_update)

    def set_datapattern(self, datapattern):
        self.bp_w.set_datapattern(datapattern)

    def title_update(self):
        if self.bp_w.are_changes_saved() is False:
            if self.window_title[-1] == "*":
                pass
            else:
                self.window_title = self.window_title + '*'
        else:
            if self.window_title[-1] == "*":
                self.window_title = self.window_title[0:-1]
        self.setWindowTitle(self.window_title)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        config.write()


class BkgPattern_widget(pyfdd.gui.datapattern_interface.DataPattern_widget):
    """ Data pattern widget class"""

    #datapattern_opened = QtCore.pyqtSignal()
    #datapattern_changed = QtCore.pyqtSignal()
    #datapattern_saved = QtCore.pyqtSignal()

    def __init__(self, *args, mainwindow=None, **kwargs):
        """
        Init method for the data pattern widget
        :param args:
        :param mainwindow: Main window object
        :param kwargs:
        """

        super(BkgPattern_widget, self).__init__(*args, mainwindow=mainwindow, **kwargs)

        # Remove widgets
        self.gridLayout_4.removeWidget(self.pb_orientchanneling)
        self.gridLayout_4.removeWidget(self.pb_editorientation)
        self.gridLayout_4.removeWidget(self.pb_fitrange)
        self.pb_orientchanneling.deleteLater()
        self.pb_editorientation.deleteLater()
        self.pb_fitrange.deleteLater()
        self.pb_orientchanneling = None
        self.pb_editorientation = None
        self.pb_fitrange = None

        # Adapt widgets
        self.dp_menu.setTitle('Background')

        # Add background tools
        self.bkgtools = BkgTools_groupbox(parent_widget=self)
        self.verticalLayout_2.insertWidget(3, self.bkgtools)

        # Popup widgets that need a reference in self
        self.dp_external = []

        # Set config section
        if not config.parser.has_section('bkgpattern'):
            config.parser.add_section('bkgpattern')

        default_is_enabled = True if not config.parser.has_option('bkgpattern', 'is_enabled') else \
            config.getlist('bkgpattern', 'is_enabled')
        default_corr_factor = 1 if not config.parser.has_option('bkgpattern', 'corr_factor') else \
            config.getlist('bkgpattern', 'corr_factor')
        default_smooth_sigma = 0 if not config.parser.has_option('bkgpattern', 'smooth_sigma') else \
            config.getlist('bkgpattern', 'smooth_sigma')

        # Variables
        self.is_enabled = default_is_enabled
        self.corr_factor = default_corr_factor
        self.smooth_sigma = default_smooth_sigma

        self.bkgtools.le_gauss_sigma.setText(str(self.smooth_sigma))
        self.bkgtools.le_correction_factor.setText(str(self.corr_factor))

        # Connect signals
        # Background tools
        self.bkgtools.cb_enabled.clicked.connect(self.call_cb_enabled)
        self.bkgtools.pb_set_factor.clicked.connect(self.call_pb_set_factor)
        self.bkgtools.pb_set_sigma.clicked.connect(self.call_pb_set_sigma)
        self.bkgtools.bp_view_background.clicked.connect(self.call_p_view_background)

    def call_cb_enabled(self):
        self.is_enabled = self.bkgtools.cb_enabled.isChecked()

    def call_pb_set_factor(self):
        if not self.datapattern is None:
            bkg_counts = self.datapattern.pattern_matrix.sum()
        else:
            bkg_counts = None

        factor_dialog = CorrFactor_dialog(parent_widget=self, corr_factor=self.corr_factor, bkg_counts=bkg_counts)
        if factor_dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.corr_factor = factor_dialog.corr_factor
            self.bkgtools.le_correction_factor.setText(str(self.corr_factor))
        else:
            # Canceled
            pass

    def call_pb_set_sigma(self):

        value, ok = QtWidgets.QInputDialog.getDouble(self, 'Gaussian Sigma',
                                                  'Set the sigma value to be used for smoothing the pattern\t\t\t',
                                                  value=self.smooth_sigma, min=0, decimals = 1)
        if ok:
            self.smooth_sigma = value

            self.bkgtools.le_gauss_sigma.setText(str(self.smooth_sigma))

            # update config
            config.parser['bkgpattern']['smooth_sigma'] = str(self.smooth_sigma)

    def call_p_view_background(self):
        if not self.datapattern_exits():
            return

        btools = pyfdd.BackgroundTools()
        btools.set_sigma(self.smooth_sigma)
        background_dp = btools.get_smoothed_background(self.datapattern, as_datapattern=True)
        new_dp_window = DataPattern_window()
        new_dp_window.set_datapattern(background_dp)
        new_dp_window.setWindowTitle('Smoothed Background')
        new_dp_window.show()
        self.dp_external.append(new_dp_window)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = BkgPattern_window()
    window.show()
    print(window.size())
    sys.exit(app.exec())


if __name__ == '__main__':
    main()