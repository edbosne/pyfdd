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

        self.maintabs.addTab(self.dp_w, 'Data Pattern')
        self.maintabs.addTab(self.se_w, 'Simulations Library')
        self.maintabs.addTab(self.fm_w, 'Fit Manager')

        self.maintabs.currentChanged.connect(self.update_fm)
        self.dp_w.datapattern_opened.connect(self.update_fm)
        self.se_w.simlibrary_opened.connect(self.update_fm)

    def get_datapattern(self):
        datapattern = self.dp_w.get_datapattern()
        return datapattern

    def get_simlibrary(self):
        simlibrary = self.se_w.get_simlibrary()
        return simlibrary

    def update_fm(self, tab=2):
        if tab == 2:  # Fit manager tab
            self.fm_w.update_all()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """ close all windows when main is closed. """
        event.accept()
        sys.exit()


def run():
    app = QtWidgets.QApplication(sys.argv)
    window = WindowedPyFDD()
    window.show()
    # print(window.size())
    sys.exit(app.exec())


if __name__ == '__main__':
    run()
