
import sys
import os
import warnings

from PyQt5 import QtCore, QtGui, QtWidgets, uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.ma as ma


import pyfdd

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
from qt_designer.datapattern_widget import Ui_DataPatternWidget
from qt_designer.buildmesh_dialog import Ui_BuildMeshDialog


# Set style
sns.set_style('white')
sns.set_context('talk')


class BuildMesh_dialog(QtWidgets.QDialog, Ui_BuildMeshDialog):
    def __init__(self, parent=None):
        super(BuildMesh_dialog, self).__init__()
        self.parent = parent
        self.setupUi(self)

        # Connect signals
        self.le_pixelsize.textEdited.connect(self.set_detector_config)
        self.le_distance.textEdited.connect(self.set_detector_config)
        self.le_angstep.textEdited.connect(self.set_step_config)

    def get_settings(self):
        settings = dict()
        settings['pixel size'] = float(self.le_pixelsize.text())
        settings['distance'] = float(self.le_distance.text())
        settings['angular step'] = float(self.le_angstep.text())
        if self.rb_detector.isChecked() and not self.rb_step.isChecked():
            settings['selected'] = 'detector'
        elif not self.rb_detector.isChecked() and self.rb_step.isChecked():
            settings['selected'] = 'step'
        else:
            warnings.warn('One and only one option must be selected')
        return settings

    def set_detector_config(self):
        self.rb_detector.setChecked(True)
        self.rb_step.setChecked(False)

    def set_step_config(self):
        self.rb_detector.setChecked(False)
        self.rb_step.setChecked(True)


class DataPattern_window(QtWidgets.QMainWindow):
    """ Class to use the data pattern widget in a separate window"""
    def __init__(self, *args, **kwargs):
        super(DataPattern_window, self).__init__(*args, **kwargs)

        # Setup the window
        self.setWindowTitle("Data Pattern")
        self.statusBar()

        # Set a DataPattern widget as central widget
        dp_w = DataPattern_widget(mainwindow=self)
        self.setCentralWidget(dp_w)
        dp_w.resize(1000, 600)


class DataPattern_widget(QtWidgets.QWidget, Ui_DataPatternWidget):
    """ Data pattern widget class"""

    def __init__(self, *args, mainwindow=None, **kwargs):
        """
        Init method for the data pattern widget
        :param args:
        :param mainwindow: Main window object
        :param kwargs:
        """

        super(DataPattern_widget, self).__init__(*args, **kwargs)

        # Alternative way to load the ui created with PyQt creator
        # uic.loadUi('qt_designer/datapattern_widget.ui', self)

        self.setupUi(self)
        self.mainwindow = mainwindow

        # initiate variables
        self.datapattern = None

        # Create a menubar entry for the datapattern
        self.menubar = self.mainwindow.menuBar()
        self.dp_menu = self.setup_menu()

        # Set up matplotlib canvas
        self.pltfig = plt.figure()
        self.pltfig.set_facecolor('#d7d6d5')
        self.plot_ax = self.pltfig.add_subplot(111)
        self.colorbar_ax = None
        self.plot_ax.set_aspect('equal')
        plt.tight_layout()
        self.addmpl(self.pltfig)
        self.maskpixel_mpl_cid = None

        # Connect signals
        self.pb_buildmesh.clicked.connect(self.call_pb_buildmesh)

        self.pb_maskpixel.clicked.connect(self.call_pb_maskpixel)
        self.pb_maskrectangle.clicked.connect(self.call_pb_maskrectangle)
        self.pb_removeedge.clicked.connect(self.call_pb_removeedge)



    def setup_menu(self):
        dp_menu = self.menubar.addMenu('&Data Pattern')

        # Open DataPattern
        open_act = QtWidgets.QAction('&Open',self)
        open_act.setStatusTip('Open a DataPattern.json file')
        open_act.triggered.connect(self.open_dp_call)
        dp_menu.addAction(open_act)

        # Open DataPattern
        openadd_act = QtWidgets.QAction('&Add', self)
        openadd_act.setStatusTip('Add DataPatterns together')
        openadd_act.triggered.connect(self.openadd_dp_call)
        dp_menu.addAction(openadd_act)

        # Import pattern matrix
        import_act = QtWidgets.QAction('&Import', self)
        import_act.setStatusTip('Import a pattern from an ASCII or .2db file')
        import_act.triggered.connect(self.import_dp_call)
        dp_menu.addAction(import_act)

        # Separate input from output
        dp_menu.addSeparator()

        # Save DataPattern
        save_act = QtWidgets.QAction('&Save json', self)
        save_act.setStatusTip('Save as a DataPattern.json file')
        save_act.triggered.connect(self.save_dp_call)
        dp_menu.addAction(save_act)

        # Export as ascii pattern matrix
        exportascii_act = QtWidgets.QAction('&Export ascii', self)
        exportascii_act.setStatusTip('Export as an ascii file')
        exportascii_act.triggered.connect(self.exportascii_dp_call)
        dp_menu.addAction(exportascii_act)

        # Export as origin pattern matrix
        exportorigin_act = QtWidgets.QAction('&Export binary', self)
        exportorigin_act.setStatusTip('Export as a binary file')
        exportorigin_act.triggered.connect(self.exportorigin_dp_call)
        dp_menu.addAction(exportorigin_act)

        # Save as image
        saveimage_act = QtWidgets.QAction('&Save Image', self)
        saveimage_act.setStatusTip('Save pattern as an image')
        saveimage_act.triggered.connect(self.saveasimage_call)
        dp_menu.addAction(saveimage_act)

        return dp_menu

    def open_dp_call(self):
        '''
        Open a json datapattern file
        :return:
        '''
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open DataPattern', filter='DataPattern (*.json)')
        if filename == ('', ''):  # Cancel
            return

        self.datapattern = pyfdd.DataPattern(filename[0])

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()

    def openadd_dp_call(self):
        '''
        Open a json datapattern file
        :return:
        '''
        filename = QtWidgets.QFileDialog.getOpenFileNames(self, 'Add DataPatterns', filter='DataPattern (*.json)')
        if filename == ('', ''):  # Cancel
            return

        for each in filename[0]:
            if self.datapattern is None:
                self.datapattern = pyfdd.DataPattern(each)
            else:
                self.datapattern = self.datapattern + pyfdd.DataPattern(each)

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()

    def save_dp_call(self):
        '''
        Save the current json file
        :return:
        '''
        if not self.datapattern_exits():
            return

        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save DataPattern', filter='DataPattern (*.json)')
        self.datapattern.io_save_json(filename[0])

    def import_dp_call(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Import matrix file',
                                                         filter='Import matrix (*.txt *.csv *.2db)')
        if filename == ('', ''):  # Cancel
            return

        import_options = ('Single chip', 'Timepix quad')
        item, ok = QtWidgets.QInputDialog.getItem(self, "Select import format",
                                                  "Import format", import_options, 0, False)

        if not ok:
            return
        elif item == 'Single chip':
            self.datapattern = pyfdd.DataPattern(file_path=filename[0], nChipsX=1, nChipsY=1, real_size=1)
        elif item == 'Timepix quad':
            self.datapattern = pyfdd.DataPattern(file_path=filename[0], nChipsX=2, nChipsY=2, real_size=3)

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()

    def exportascii_dp_call(self):
        if not self.datapattern_exits():
            return

        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Export DataPattern', filter='ASCII (*.txt)')
        self.datapattern.io_save_ascii(filename[0])

    def exportorigin_dp_call(self):
        if not self.datapattern_exits():
            return

        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Export DataPattern', filter='Binary (*.2db)')
        self.datapattern.io_save_origin(filename[0])

    def saveasimage_call(self):
        if not self.datapattern_exits():
            return



        filename = QtWidgets.QFileDialog.\
            getSaveFileName(self, 'Export DataPattern',
                            filter='image (*emf *eps *.pdf *.png *.ps *.raw *.rgba *.svg *.svgz)')

        # Save with a white background
        #self.pltfig.set_facecolor('white')
        self.pltfig.savefig(filename[0], dpi=600, facecolor='white')
        #self.pltfig.set_facecolor('#d7d6d5')
        #self.canvas.draw()

    def datapattern_exits(self):
        """
        Warn if the datapattern is None
        :return: bool
        """
        if self.datapattern is None:
            QtWidgets.QMessageBox.warning(self,'Warning message','The DataPattern does not exist.')
            return False
        else:
            return True

    def draw_datapattern(self):

        # Clear previous axes and colorbar

        if self.colorbar_ax is not None:
            self.colorbar_ax.remove()
        if self.plot_ax is not None:
            self.plot_ax.clear()

        self.plot_ax, self.colorbar_ax = \
            self.datapattern.draw(self.plot_ax, percentiles=(0.04, 0.99), title='', xlabel='', ylabel='', zlabel='')
        # The reason why the plot moves is because of the tight_layout
        # plt.tight_layout()
        self.canvas.draw()
        self.plot_ax.set_aspect('equal')

    def on_move(self,event):
        #print(event)
        if event.inaxes == self.plot_ax:
            x, y = event.xdata, event.ydata
            if self.datapattern is not None:
                i,j = self.get_index_from_xy(x,y)
                z = self.datapattern.matrixDrawable[i, j]
            else:
                z = 0

            self.mainwindow.statusBar().showMessage('(x,y,z) - ({:.2},{:.2},{})'.format(x, y, z))
        else:
            self.mainwindow.statusBar().showMessage('')

    def get_index_from_xy(self,x,y):
        xm = self.datapattern.xmesh[0,:]
        ym = self.datapattern.ymesh[:,0]

        j = (np.abs(xm - x)).argmin() #columns
        i = (np.abs(ym - y)).argmin() #lines

        return i,j

    def addmpl(self, fig):

        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()

        # connect status bar coordinates display
        self.canvas.mpl_connect('motion_notify_event', self.on_move)

    def update_infotext(self):
        base_text = 'Total counts: {:.1f}; Valid: {:.1f}\n' \
                    'Active pixels: {:d}; Masked: {:d}\n' \
                    'Angular range (x, y): {:.1f}, {:.1f}\n' \
                    'Pattern orientation (x, y, phi): {:.2f}, {:.2f}, {:.1f}'

        total_c = self.datapattern.matrixCurrent.data.sum()
        total_c_active = self.datapattern.matrixCurrent.sum()
        masked_p = self.datapattern.matrixCurrent.mask.sum()
        active_p = (~self.datapattern.matrixCurrent.mask).sum()
        xm = self.datapattern.xmesh[0, :]
        ym = self.datapattern.ymesh[:, 0]
        x_range = xm[-1] - xm[0]
        y_range = ym[-1] - ym[0]
        x_orient, y_orient = self.datapattern.center
        phi = self.datapattern.angle

        text = base_text.format(total_c, total_c_active, active_p, masked_p, x_range, y_range, x_orient, y_orient, phi)

        self.infotext.setText(text)

    def call_pb_buildmesh(self):

        if not self.datapattern_exits():
            return

        buildmesh_dialog = BuildMesh_dialog(parent=self)

        if buildmesh_dialog.exec_() == QtWidgets.QDialog.Accepted:
            output = buildmesh_dialog.get_settings()
            if output['selected'] == 'detector':
                self.datapattern.manip_create_mesh(pixel_size=output['pixel size'],
                                                   distance=output['distance'])
            elif output['selected'] == 'step':
                dummy_distance = 1000
                dummy_pixel_size = np.tan(output['angular step'] * np.pi / 180) * dummy_distance
                self.datapattern.manip_create_mesh(pixel_size=dummy_pixel_size,
                                                   distance=dummy_distance)
            else:
                warnings.warn('Non valid selection')
        else:
            pass
            #print('Cancelled')

        buildmesh_dialog.deleteLater()

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()

    def on_maskpixelclick(self, event):
        if event.button == 1:
            x, y = event.xdata, event.ydata
            i, j = self.get_index_from_xy(x, y)
            self.datapattern.mask_pixel(i, j)

            # Draw pattern and update info text
            self.draw_datapattern()
            self.update_infotext()

    def call_pb_maskpixel(self):
        if not self.datapattern_exits():
            return

        if self.pb_maskpixel.isChecked():
            self.maskpixel_mpl_cid = self.canvas.mpl_connect('button_press_event', self.on_maskpixelclick)
        else:
            self.canvas.mpl_disconnect(self.maskpixel_mpl_cid)

    def on_rectangleselect(self, eclick, erelease):
        # eclick and erelease are matplotlib events at press and release
        rectangle_limits = np.array([eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata])

        self.datapattern.mask_rectangle(rectangle_limits)

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()

    def call_pb_maskrectangle(self):
        if not self.datapattern_exits():
            return

        if self.pb_maskrectangle.isChecked():
            rectprops = dict(facecolor='red', edgecolor='black',
                             alpha=0.8, fill=True)
            # useblit=True is necessary for PyQt
            self.RS = RectangleSelector(self.plot_ax, self.on_rectangleselect, drawtype='box', useblit=True, interactive=False,
                                        rectprops=rectprops)
        else:
            self.RS = None

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()

    def call_pb_removeedge(self):

        if not self.datapattern_exits():
            return

        value, ok = QtWidgets.QInputDialog.getInt(self, 'Input value', 'Number of edge pixels to remove\t\t\t',# 0,0)
                                                  value=0, min=0)
        print(value, ok)

        # Draw pattern and update info text
        self.draw_datapattern()
        self.update_infotext()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    #window = DataPattern_widget()
    window = DataPattern_window()
    window.show()
    print(window.size())
    sys.exit(app.exec())
