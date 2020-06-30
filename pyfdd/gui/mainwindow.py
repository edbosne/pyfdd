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

        self.maintabs.addTab(self.dp_w, 'Data Pattern')
        self.maintabs.addTab(self.se_w, 'Simulations Library')

        pyqt_bkg = self.maintabs.palette().color(QtGui.QPalette.Base).getRgbF()
        mpl_bkg = mpl.colors.rgb2hex(pyqt_bkg)
        mpl_bkg = '#fcfcfc' # couldn't get the exact color so I got this value by hand

        self.dp_w.dpcontroler.refresh_mpl_color(new_mpl_bkg=mpl_bkg)
        self.se_w.dpcontroler.refresh_mpl_color(new_mpl_bkg=mpl_bkg)


def run():
    app = QtWidgets.QApplication(sys.argv)
    window = WindowedPyFDD()
    window.show()
    print(window.size())
    sys.exit(app.exec())

if __name__ == '__main__':
    run()