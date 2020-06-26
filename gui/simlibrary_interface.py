
import sys
import os
import warnings

from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from PySide2 import QtCore, QtGui, QtWidgets, uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT #as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from pyfdd.datapattern.CustomWidgets import AngleMeasure
import matplotlib.pyplot as plt  # do not use pyplot
import matplotlib as mpl
import seaborn as sns
import numpy as np

import pyfdd

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from gui.qt_designer.simexplorer_widget import Ui_SimExplorerWidget
from gui.datapattern_interface import DataPatternControler


class SimExplorer_window(QtWidgets.QMainWindow):
    """ Class to use the data pattern widget in a separate window"""
    def __init__(self, *args, **kwargs):
        super(SimExplorer_window, self).__init__(*args, **kwargs)

        # Setup the window
        self.setWindowTitle("Data Pattern")
        self.statusBar()

        # Set a DataPattern widget as central widget
        dp_w = SimExplorer_widget(mainwindow=self)
        self.setCentralWidget(dp_w)
        self.resize(1150, 670)


class SimExplorer_widget(QtWidgets.QWidget, Ui_SimExplorerWidget):
    """ Data pattern widget class"""

    def __init__(self, *args, mainwindow=None, **kwargs):
        """
        Init method for the data pattern widget
        :param args:
        :param mainwindow: Main window object
        :param kwargs:
        """

        super(SimExplorer_widget, self).__init__(*args, **kwargs)

        # Alternative way to load the ui created with PyQt creator
        # uic.loadUi('qt_designer/datapattern_widget.ui', self)

        self.setupUi(self)
        self.mainwindow = mainwindow

        # Instantiate datapattern controler
        self.dpcontroler = DataPatternControler(parent_widget=self, mpl_layout=self.mplvl, infotext_box=None)
        self.dpcontroler.percentiles = [0, 1]

        # Create a menubar entry for the datapattern
        self.menubar = self.mainwindow.menuBar()
        self.dp_menu = self.setup_menu()

        # Variables
        self.current_row = None

        # Connect signals
        # List
        self.simlist.currentItemChanged.connect(self.update_datapattern)

        # Pattern visualization
        self.pb_colorscale.clicked.connect(self.dpcontroler.call_pb_colorscale)
        self.pb_setlabels.clicked.connect(self.dpcontroler.call_pb_setlabels)

    def setup_menu(self):
        dp_menu = self.menubar.addMenu('&Sim. Library')

        # Open 2dl
        open_act = QtWidgets.QAction('&Open', self)
        open_act.setStatusTip('Open a .2dl simulations file')
        open_act.triggered.connect(self.open_2dl_call)
        dp_menu.addAction(open_act)

        # Separate input from output
        dp_menu.addSeparator()

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

        return dp_menu

    def open_2dl_call(self):
        """
        Open a 2dl library file
        :return:
        """
        lib_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open .2dl library', filter='library (*.2dl)',
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if lib_path == ('', ''):  # Cancel
            return

        self.simlib = pyfdd.Lib2dl(lib_path[0])

        # update info text, simlist and datapattern
        self.update_infotext()
        self.update_simlist()
        self.update_datapattern()

    def update_infotext(self):
        if self.infotext is None:
            #raise warnings.warn('Info text box is not set')
            return

        base_text = 'Pattern dimentions (nx, ny): {:d}, {:d}\n' \
                    'Angular step (x, y): {:.2f}, {:.2f}\n' \
                    'Angular range (x, y): {:.2f}, {:.2f}\n' \
                    'Number of simulations: {:d}'
        dict_2dl = self.simlib.dict_2dl
        nx, ny = self.simlib.nx_mirror, self.simlib.ny_mirror
        xstep, ystep = self.simlib.xstep, self.simlib.ystep
        xfirst, yfirst = self.simlib.xfirst, self.simlib.yfirst
        xlast, ylast = self.simlib.xlast, self.simlib.ylast
        xrange = xlast - xfirst
        yrange = ylast - yfirst
        num_sim = len(dict_2dl['Spectrums'])

        text = base_text.format(nx, ny, xstep, ystep, xrange, yrange, num_sim)

        self.infotext.setText(text)

    def update_simlist(self):
        baseline_string = '{:>3} - {:25}, f: {:}, u1: {:}, s:{:}'
        for line in self.simlib.get_simulations_list():
            # columns, ["Spectrum number", "Spectrum_description", "factor", "u1", "sigma"]
            # strip white spaces from description
            line_string = baseline_string.format(line[0],line[1].strip(),*line[2:])
            self.simlist.addItem(line_string)

        self.current_row = 1
        self.simlist.setCurrentRow(self.current_row - 1)

    def update_datapattern(self):
        self.current_row = self.simlist.currentRow() + 1
        self.dpcontroler.set_datapattern(self.simlib.get_simulation_patt_as_dp(self.current_row))


def main():
    app = QtWidgets.QApplication(sys.argv)
    # window = DataPattern_widget()
    window = SimExplorer_window()
    window.show()
    print(window.size())
    sys.exit(app.exec())

if __name__ == '__main__':
    main()