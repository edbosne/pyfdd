import sys
import os
import warnings

from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from PySide2 import QtCore, QtGui, QtWidgets, uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT #as NavigationToolbar
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
from pyfdd.gui.datapattern_interface import DataPatternControler
import pyfdd.gui.config as config


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

        # Variables


        # Connect signals
        # Control

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


def main():
    app = QtWidgets.QApplication(sys.argv)
    # window = DataPattern_widget()
    window = PatternCreator_window()
    window.show()
    print(window.size())
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
