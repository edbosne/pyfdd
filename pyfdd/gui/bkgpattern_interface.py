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

        #
        self.bkgtools = BkgTools_groupbox(parent_widget=self)
        self.verticalLayout_2.insertWidget(3, self.bkgtools)




def main():
    app = QtWidgets.QApplication(sys.argv)
    window = BkgPattern_window()
    window.show()
    print(window.size())
    sys.exit(app.exec())


if __name__ == '__main__':
    main()