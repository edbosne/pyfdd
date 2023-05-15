import sys
import os
# import warnings

from PyQt5 import QtCore, QtGui, QtWidgets
# from PySide2 import QtCore, QtGui, QtWidgets, uic

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT #as NavigationToolbar
# from matplotlib.widgets import RectangleSelector
# from pyfdd.core.datapattern.CustomWidgets import AngleMeasure
# import matplotlib.pyplot as plt  # do not use pyplot
# import matplotlib as mpl
# import seaborn as sns
# import numpy as np

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from pyfdd.gui.qt_designer.windowedpyfdd import Ui_WindowedPyFDD
from pyfdd.gui.datapattern_interface import DataPattern_widget
from pyfdd.gui.simlibrary_interface import SimExplorer_widget
from pyfdd.gui.fitmanager_interface import FitManager_widget
from pyfdd.gui.patterncreator_interface import PatternCreator_widget
from pyfdd.gui.bkgpattern_interface import BkgPattern_widget
import pyfdd.gui.config as config


class VLine(QtWidgets.QFrame):
    """
    A simple VLine, like the one you get from designer
    """
    def __init__(self):
        super(VLine, self).__init__()
        self.setFrameShape(self.VLine | self.Sunken)


class WindowedPyFDD(QtWidgets.QMainWindow, Ui_WindowedPyFDD):
    """ Class to use the data pattern widget in a separate window"""
    def __init__(self, *args, **kwargs):
        """
        Init method for the windowed PyFDD
        :param args:
        :param kwargs:
        """
        # Set up the window
        super(WindowedPyFDD, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # Add a status bar
        self.sb_datapattern_name = None
        self.sb_lib2dl_name = None
        self.setup_statusbar()

        # Load configuration
        config.filename = 'pyfdd_config.ini'
        self.config_load()

        # Create pyfdd widgets
        self.dp_w = DataPattern_widget(self.maintabs, mainwindow=self)
        self.se_w = SimExplorer_widget(self.maintabs, mainwindow=self)
        self.fm_w = FitManager_widget(self.maintabs, mainwindow=self)
        self.pc_w = PatternCreator_widget(self.maintabs, mainwindow=self)
        self.bp_w = BkgPattern_widget(self.maintabs, mainwindow=self)

        # Creat the tabs for the widgets
        self.dp_tab_title = 'Data Pattern'
        self.se_tab_title = 'Simulations Library'
        self.fm_tab_title = 'Fit Manager'
        self.pc_tab_title = 'Pattern Creator'
        self.bp_tab_title = 'Background Pattern'

        self.maintabs.addTab(self.dp_w, 'Data Pattern')
        self.maintabs.addTab(self.se_w, 'Simulations Library')
        self.maintabs.addTab(self.fm_w, 'Fit Manager')
        self.maintabs.addTab(self.pc_w, 'Pattern Creator')
        self.maintabs.addTab(self.bp_w, 'Background Pattern')

        # Connect signals
        # Update fit manager tab if data pattern or library is changed
        self.fm_pending_update = False  # Updates when switching to its tab
        self.maintabs.currentChanged.connect(self.update_fm)
        self.dp_w.datapattern_opened.connect(lambda: self._set_fm_pending_update(True))
        self.dp_w.datapattern_changed.connect(lambda: self._set_fm_pending_update(True))
        self.se_w.simlibrary_opened.connect(lambda: self._set_fm_pending_update(True))

        # Update pattern creator tab if data pattern or library is changed
        self.pc_pending_update = False  # Updates when switching to its tab
        self.maintabs.currentChanged.connect(self.update_pc)
        self.dp_w.datapattern_opened.connect(lambda: self._set_pc_pending_update(True))
        self.dp_w.datapattern_changed.connect(lambda: self._set_pc_pending_update(True))
        self.se_w.simlibrary_opened.connect(lambda: self._set_pc_pending_update(True))

        # Update the filename
        # Data pattern
        self.dp_w.datapattern_opened.connect(self.update_dp_filename)
        self.dp_w.datapattern_saved.connect(self.update_dp_filename)
        # Sim explorer
        self.se_w.simlibrary_opened.connect(self.update_lib_filename)

        # Add a star to the tab title if the tab is not saved
        # Data pattern
        self.dp_w.datapattern_opened.connect(self.dp_tab_title_update)
        self.dp_w.datapattern_changed.connect(self.dp_tab_title_update)
        self.dp_w.datapattern_saved.connect(self.dp_tab_title_update)
        # Fit manager
        self.fm_w.fitresults_changed.connect(self.fm_tab_title_update)
        self.fm_w.fitresults_saved.connect(self.fm_tab_title_update)
        # Background pattern
        self.bp_w.datapattern_opened.connect(self.bp_tab_title_update)
        self.bp_w.datapattern_changed.connect(self.bp_tab_title_update)
        self.bp_w.datapattern_saved.connect(self.bp_tab_title_update)

    def setup_statusbar(self):
        """
        Add the status bar and set it up properly.
        """
        # Status bar
        self.statusBar()

        # Vertical line
        self.statusBar().addPermanentWidget(VLine())
        # Label with datapattern filename
        self.sb_datapattern_name = QtWidgets.QLabel('Data Pattern file : ', parent=self)
        self.statusBar().addPermanentWidget(self.sb_datapattern_name)

        # Vertical line
        self.statusBar().addPermanentWidget(VLine())
        # Label with library filename
        self.sb_lib2dl_name = QtWidgets.QLabel('Library file : ', parent=self)
        self.statusBar().addPermanentWidget(self.sb_lib2dl_name)

    def update_dp_filename(self):
        """
        Set the Data Pattern filename on the status bar.
        :return:
        """
        filename = os.path.basename(self.dp_w.dp_filename)
        assert isinstance(self.sb_datapattern_name, QtWidgets.QLabel)
        self.sb_datapattern_name.setText('Data Pattern file : {}'.format(filename))

    def update_lib_filename(self):
        """
        Set the 2dl library filename on the status bar.
        :return:
        """
        filename = os.path.basename(self.se_w.simlib_filename)
        assert isinstance(self.sb_lib2dl_name, QtWidgets.QLabel)
        self.sb_lib2dl_name.setText('Library file : {}'.format(filename))

    def _set_fm_pending_update(self, pending: bool):
        """
        Set True is a pending update is missing for the fit manager.
        :param pending: Boolean value to set the pending update.
        :return:
        """
        self.fm_pending_update = pending

        # If the fit manager tab is already open then update now
        if self.maintabs.currentIndex() == 2:
            self.update_fm()

    def _set_pc_pending_update(self, pending: bool):
        """
        Set True is a pending update is missing for the pattern creator.
        :param pending: Boolean value to set the pending update.
        :return:
        """
        self.pc_pending_update = pending

        # If the fit manager tab is already open then update now
        if self.maintabs.currentIndex() == 3:
            self.update_pc()

    def get_datapattern(self):
        """
        Get the DataPattern object from its tab widget.
        :return:
        """
        datapattern = self.dp_w.get_datapattern()
        return datapattern

    def get_bkgtab_datapattern(self):
        """
        Get the DataPattern object from its tab widget.
        :return:
        """
        datapattern = self.bp_w.get_datapattern()
        return datapattern

    def get_simlibrary(self):
        """
        Get the Lib2dl object from its tab widget.
        :return:
        """
        simlibrary = self.se_w.get_simlibrary()
        return simlibrary

    def get_background_pattern_and_factor(self):
        return self.bp_w.get_background_pattern_and_corrfactor()

    def update_fm(self, tab=2):
        """
        Update the fit manager.
        :param tab: Tab index. Fit manager is expected on tab == 2
        :return:
        """
        if tab == 2 and self.fm_pending_update:  # Fit manager tab == 2
            self.fm_w.update_all()
            self._set_fm_pending_update(False)

    def update_pc(self, tab=3):
        """
        Update the pattern creator.
        :param tab: Tab index. Pattern creator is expected on tab == 3
        :return:
        """
        if tab == 3 and self.pc_pending_update:  # Pattern creator tab == 3
            self.pc_w.update_all()
            self._set_pc_pending_update(False)

    def dp_tab_title_update(self):
        """
        Update the title of the datapattern tab if the saved status changes.
        :return:
        """
        if self.dp_w.are_changes_saved() is False:
            if self.dp_tab_title[-1] == "*":
                pass
            else:
                self.dp_tab_title = self.dp_tab_title + '*'
        else:
            if self.dp_tab_title[-1] == "*":
                self.dp_tab_title = self.dp_tab_title[0:-1]

        self.maintabs.setTabText(0, self.dp_tab_title)  # DP tab index is 0

    def fm_tab_title_update(self):
        """
        Update the title of the fit manager tab if the saved status changes.
        :return:
        """
        if self.fm_w.are_changes_saved() is False:
            if self.fm_tab_title[-1] == "*":
                pass
            else:
                self.fm_tab_title = self.fm_tab_title + '*'
        else:
            if self.fm_tab_title[-1] == "*":
                self.fm_tab_title = self.fm_tab_title[0:-1]

        self.maintabs.setTabText(2, self.fm_tab_title)  # FM tab index is 2

    def bp_tab_title_update(self):
        """
        Update the title of the Background Pattern tab if the saved status changes.
        :return:
        """
        if self.bp_w.are_changes_saved() is False:
            if self.bp_tab_title[-1] == "*":
                pass
            else:
                self.bp_tab_title = self.bp_tab_title + '*'
        else:
            if self.bp_tab_title[-1] == "*":
                self.bp_tab_title = self.bp_tab_title[0:-1]

        self.maintabs.setTabText(4, self.bp_tab_title)  # BP tab index is 4

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """
        Close all windows when main is closed.
        :param event: A QtGui close event
        :return:
        """

        quit_msg = "Are you sure you want to exit the program?"
        if self.dp_w.are_changes_saved() is False or self.fm_w.are_changes_saved() is False:
            quit_msg = quit_msg + '\n\nAtention:'

        if self.dp_w.are_changes_saved() is False:
            quit_msg = quit_msg + '\n  - Data Pattern is not saved!'
        if self.fm_w.are_changes_saved() is False:
            quit_msg = quit_msg + '\n  - Fit results are not saved!'

        reply = QtWidgets.QMessageBox.question(self, 'Message',
                                               quit_msg, QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Cancel)

        if reply == QtWidgets.QMessageBox.Close:
            config.write()
            event.accept()
            sys.exit()
        else:
            event.ignore()

    @staticmethod
    def config_load():
        """
        Load the configuration file.
        :return:
        """

        config.read()

        # Load variables
        if 'pyfdd' not in config.parser:
            config.parser['pyfdd'] = dict()

        # At the moment the main window does not have any variables to load.


def run():
    """
    Main application run method.
    :return:
    """
    app = QtWidgets.QApplication(sys.argv)
    window = WindowedPyFDD()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run()
