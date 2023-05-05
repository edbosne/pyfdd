
import sys
import os
import warnings
from enum import Enum, IntEnum


from PyQt5 import QtCore, QtGui, QtWidgets, uic, sip
# from PySide2 import QtCore, QtGui, QtWidgets, uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import pandas.plotting._matplotlib  # Necessary import to avoid crashes on windows with pyinstaller

import pyfdd

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from pyfdd.gui.pandasmodel import DataFrameModel
from pyfdd.gui.qt_designer.viewresults_widget import Ui_ViewResultsWidget
import pyfdd.gui.config as config


class ViewResults_widget(QtWidgets.QWidget, Ui_ViewResultsWidget):
    """ Data pattern widget class"""

    def __init__(self, *args, fitman_output:pyfdd.FitManager, parent_widget=None, **kwargs):
        """
        Init method for the view results widget
        :param args:
        :param mainwindow: Main window object
        :param kwargs:
        """

        if not isinstance(fitman_output, pyfdd.FitManager):
            raise TypeError(f'Argument fitman_output should be of type pyfdd.FitManager and not {type(fitman_output)}')

        super(ViewResults_widget, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # Set config section
        if not config.parser.has_section('viewresults'):
            config.parser.add_section('viewresults')

        # Load or create variables
        self.simplify_table_view = False if not config.parser.has_option('viewresults', 'simplify_table_view') else \
            config.getboolean('viewresults', 'simplify_table_view')

        self.fitman = fitman_output
        self.results_df = self.fitman.results.generate_results_table(layout='horizontal',
                                                                       simplify =self.simplify_table_view)
        self.parent_widget = parent_widget

        # set the mpl widget background colour
        self.mplframe.setStyleSheet('background: palette(window);')

        # Set up matplotlib canvas
        # get background color from widget and convert it to RBG
        pyqt_bkg = self.parent_widget.mainwindow.palette().color(QtGui.QPalette.Background).getRgbF()
        mpl_bkg = mpl.colors.rgb2hex(pyqt_bkg)

        # self.pltfig = plt.figure() # don't use pyplot
        # print(dir(mpl.figure))
        self.pltfig_costfunc = mpl.figure.Figure(dpi=60, figsize=[1.5*6.4, 4.8])
        self.pltfig_fractions = mpl.figure.Figure(dpi=60, figsize=[1.5*6.4, 4.8])
        self.pltfig_costfunc .set_facecolor(mpl_bkg)
        self.pltfig_fractions.set_facecolor(mpl_bkg)
        self.plot_ax_costfunc = self.pltfig_costfunc.add_subplot(111)
        self.plot_ax_fractions = self.pltfig_fractions.add_subplot(111)

        # Add canvas widgets
        self.mpl_canvas_costfunc = FigureCanvas(self.pltfig_costfunc)
        self.mpl_hlayout.addWidget(self.mpl_canvas_costfunc)
        self.mpl_canvas_costfunc.draw()
        self.mpl_canvas_fractions = FigureCanvas(self.pltfig_fractions)
        self.mpl_hlayout.addWidget(self.mpl_canvas_fractions)
        self.mpl_canvas_fractions.draw()

        # Connect signals
        self.pb_simplify.clicked.connect(self.call_pb_simplify_table)

        self.update_pb_simplify()
        self.show_table()
        self.show_plots()

    def show_plots(self):
        # value
        self.results_df.plot(y='value', ax=self.plot_ax_costfunc, legend=False)
        self.plot_ax_costfunc.set_xlabel('Fit number')
        self.plot_ax_costfunc.set_ylabel('Value')
        self.plot_ax_costfunc.set_title('Cost Function')

        # site1 fraction
        columns = []
        for col in self.results_df.columns:
            if 'site' in col and 'fraction' in col:
                columns.append(col)
        self.results_df.plot(y=columns, ax=self.plot_ax_fractions, legend=True)
        self.plot_ax_fractions.set_xlabel('Fit number')
        self.plot_ax_fractions.set_ylabel('Fraction')
        self.plot_ax_fractions.set_title('Site fractions')

        # call tight_layout after addmpl
        self.pltfig_costfunc.tight_layout()
        self.pltfig_fractions.tight_layout()

    def show_table(self):
        model = DataFrameModel(self.results_df)
        self.ResultsTable.setModel(model)
        self.ResultsTable.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.ResultsTable.resizeColumnsToContents()

    def call_pb_simplify_table(self):
        self.simplify_table_view = not self.simplify_table_view
        self.results_df = self.fitman.results.generate_results_table(layout='horizontal',
                                                                     simplify=self.simplify_table_view)
        self.update_pb_simplify()
        self.show_table()
        # update config
        config.parser['viewresults']['simplify_table_view'] = str(self.simplify_table_view)

    def update_pb_simplify(self):
        if self.simplify_table_view:
            self.pb_simplify.setText("Advanced Table View")
        else:
            self.pb_simplify.setText("Simplified Table View")