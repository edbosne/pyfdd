import sys
import os
import warnings

from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from PySide2 import QtCore, QtGui, QtWidgets, uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT #as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from pyfdd.core.datapattern.CustomWidgets import AngleMeasure
import matplotlib.pyplot as plt  # do not use pyplot
import matplotlib as mpl
import seaborn as sns
import numpy as np

import pyfdd

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from pyfdd.gui.qt_designer.windowedpyfdd import Ui_WindowedPyFDD
from pyfdd.gui.datapattern_interface import DataPattern_widget
from pyfdd.gui.simlibrary_interface import SimExplorer_widget
from pyfdd.gui.fitmanager_interface import FitManager_widget


class WindowedPyFDD(QtWidgets.QMainWindow, Ui_WindowedPyFDD):
    """ Class to use the data pattern widget in a separate window"""
    def __init__(self, *args, **kwargs):
        super(WindowedPyFDD, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # Setup the window
        self.statusBar()

        # Create pyfdd widgets
        self.dp_w = DataPattern_widget(self.maintabs, mainwindow=self)
        self.se_w = SimExplorer_widget(self.maintabs, mainwindow=self)
        self.fm_w = FitManager_widget(self.maintabs, mainwindow=self)

        self.dp_tab_title = 'Data Pattern'
        self.se_tab_title = 'Simulations Library'
        self.fm_tab_title = 'Fit Manager'

        self.maintabs.addTab(self.dp_w, 'Data Pattern')
        self.maintabs.addTab(self.se_w, 'Simulations Library')
        self.maintabs.addTab(self.fm_w, 'Fit Manager')

        self.maintabs.currentChanged.connect(self.update_fm)
        self.dp_w.datapattern_opened.connect(self.update_fm)
        self.se_w.simlibrary_opened.connect(self.update_fm)

        self.dp_w.datapattern_changed_or_saved.connect(self.dp_tab_title_update)
        self.fm_w.fitresults_changed_or_saved.connect(self.fm_tab_title_update)

    def get_datapattern(self):
        datapattern = self.dp_w.get_datapattern()
        return datapattern

    def get_simlibrary(self):
        simlibrary = self.se_w.get_simlibrary()
        return simlibrary

    def update_fm(self, tab=2):
        if tab == 2:  # Fit manager tab
            self.fm_w.update_all()

    def dp_tab_title_update(self):
        if self.dp_w.are_changes_saved() is False:
            if self.dp_tab_title[-1] == "*":
                pass
            else:
                self.dp_tab_title = self.dp_tab_title + '*'
        else:
            if self.dp_tab_title[-1] == "*":
                self.dp_tab_title = self.dp_tab_title[0:-1]

        self.maintabs.setTabText(0, self.dp_tab_title) # DP index is 0

    def fm_tab_title_update(self):
        if self.fm_w.are_changes_saved() is False:
            if self.fm_tab_title[-1] == "*":
                pass
            else:
                self.fm_tab_title = self.fm_tab_title + '*'
        else:
            if self.fm_tab_title[-1] == "*":
                self.fm_tab_title = self.fm_tab_title[0:-1]

        self.maintabs.setTabText(2, self.fm_tab_title) # FM index is 2

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """ close all windows when main is closed. """

        quit_msg = "Are you sure you want to exit the program?"
        if self.dp_w.are_changes_saved() is False or \
            self.fm_w.are_changes_saved() is False:
            quit_msg = quit_msg + '\n\nAtention:'

        if self.dp_w.are_changes_saved() is False:
            quit_msg = quit_msg + '\n  - Data Pattern is not saved!'
        if self.fm_w.are_changes_saved() is False:
            quit_msg = quit_msg + '\n  - Fit results are not saved!'

        reply = QtWidgets.QMessageBox.question(self, 'Message',
                                           quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            sys.exit()
        else:
            event.ignore()


def run():
    app = QtWidgets.QApplication(sys.argv)
    window = WindowedPyFDD()
    window.show()
    # print(window.size())
    sys.exit(app.exec())


if __name__ == '__main__':
    run()
