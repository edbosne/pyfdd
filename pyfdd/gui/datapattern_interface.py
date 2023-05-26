
import sys

from PyQt5 import QtCore, QtGui, QtWidgets #, uic
# from PySide2 import QtCore, QtGui, QtWidgets, uic


# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from pyfdd.gui.qt_designer.datapattern_widget import Ui_DataPatternWidget
from pyfdd.gui.datapattern_controler import DataPatternControler
import pyfdd.gui.config as config


class DataPattern_window(QtWidgets.QMainWindow):
    """ Class to use the data pattern widget in a separate window"""
    def __init__(self, *args, **kwargs):
        super(DataPattern_window, self).__init__(*args, **kwargs)

        # Load configuration
        if config.parser is None:
            config.filename = 'datapatter_config.ini'
            config.read()

        # Set up the window
        self.window_title = "Data Pattern"
        self.setWindowTitle(self.window_title)
        self.statusBar()

        # Set a DataPattern widget as central widget
        self.dp_w = DataPattern_widget(mainwindow=self)
        self.setCentralWidget(self.dp_w)
        self.resize(1150, 670)

        # Connect signals
        self.dp_w.datapattern_changed.connect(self.title_update)
        self.dp_w.datapattern_saved.connect(self.title_update)

    def set_datapattern(self, datapattern):
        self.dp_w.set_datapattern(datapattern)

    def title_update(self):
        if self.dp_w.are_changes_saved() is False:
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


class DataPattern_widget(QtWidgets.QWidget, Ui_DataPatternWidget, DataPatternControler):
    """ Data pattern widget class"""

    datapattern_opened = QtCore.pyqtSignal()
    datapattern_changed = QtCore.pyqtSignal()
    datapattern_saved = QtCore.pyqtSignal()

    def __init__(self, *args, mainwindow=None, **kwargs):
        """
        Init method for the data pattern widget
        :param args:
        :param mainwindow: Main window object
        :param kwargs:
        """

        # Alternative way to load the ui created with PyQt creator
        # uic.loadUi('qt_designer/datapattern_widget.ui', self)
        #super(QtWidgets.QWidget, self).__init__()
        super(DataPattern_widget, self).__init__(parent_widget=self, **kwargs)


        #super(DataPattern_widget, self).__init__(*args, parent_widget=self, **kwargs)

        self.setupUi(self)
        self.mainwindow = mainwindow
        self.mpl_layout = self.mplvl #super(Ui_DataPatternWidget, self).mpl_layout
        self.set_widgets_and_layouts(mainwindow=self.mainwindow, infotext_box=self.infotext, mpl_layout=self.mpl_layout)


        # set the mpl widget background colour
        self.mplwindow.setStyleSheet('background: palette(window);')

        # Instantiate datapattern controler
        # self.dpcontroler = DataPatternControler(parent_widget=self, mpl_layout=self.mplvl, infotext_box=self.infotext)
        # self.dpcontroler.datapattern_opened.connect(self.datapattern_opened.emit)
        # self.dpcontroler.datapattern_changed.connect(self.datapattern_changed.emit)
        # self.dpcontroler.datapattern_saved.connect(self.datapattern_saved.emit)

        # Create a menubar entry for the datapattern
        self.menubar = self.mainwindow.menuBar()
        self.dp_menu = self.setup_menu()

        # Variables
        self.dp_filename = ''

        # Connect signals
        # Pattern manipulation
        self.pb_buildmesh.clicked.connect(self.call_pb_buildmesh)
        self.pb_compressmesh.clicked.connect(self.call_pb_compressmesh)
        self.pb_orientchanneling.clicked.connect(lambda:self.call_pb_orientchanneling(self.pb_orientchanneling))
        self.pb_editorientation.clicked.connect(self.call_pb_editorientation)
        self.pb_fitrange.clicked.connect(self.call_pb_angularfitrange)

        # Mask signals
        # bg_mask_group can not be excluse in order to allow control over the toggled buttons.
        self.bg_mask_group.setExclusive(False)
        self.bg_mask_group.buttonClicked.connect(self.untoggle_other_pb)
        self.bg_manip_group.buttonClicked.connect(self.untoggle_other_pb)
        self.bg_vis_group.buttonClicked.connect(self.untoggle_other_pb)
        self.pb_maskpixel.clicked.connect(lambda: self.call_pb_maskpixel(self.pb_maskpixel))
        self.pb_maskrectangle.clicked.connect(lambda: self.call_pb_maskrectangle(self.pb_maskrectangle))
        self.bg_mask_group.buttonClicked.connect(self.untoggle_other_pb)
        self.pb_maskbelow.clicked.connect(self.call_pb_maskbelow)
        self.pb_maskabove.clicked.connect(self.call_pb_maskabove)
        self.pb_maskedge.clicked.connect(self.call_pb_maskedge)
        self.pb_removecentral.clicked.connect(self.call_pb_removecentral)
        self.pb_expandmask.clicked.connect(self.call_pb_expandmask)
        self.pb_clearmask.clicked.connect(self.call_pb_clearmask)
        self.pb_loadmask.clicked.connect(self.call_pb_loadmask)
        self.pb_savemask.clicked.connect(self.call_pb_savemask)

        # Pattern visualization
        self.pb_colorscale.clicked.connect(self.call_pb_colorscale)
        self.pb_setlabels.clicked.connect(self.call_pb_setlabels)

    def setup_menu(self):
        dp_menu = self.menubar.addMenu('&Data Pattern')

        # Open DataPattern
        open_act = QtWidgets.QAction('&Open', self)
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

        # Process data to impotrt pattern matrix
        processdata_act = QtWidgets.QAction('&Process data', self)
        processdata_act.setStatusTip('Process data into a 2D pattern')
        processdata_act.triggered.connect(self.processdata_dp_call)
        dp_menu.addAction(processdata_act)

        # Separate input from output
        dp_menu.addSeparator()

        # Save DataPattern
        save_act = QtWidgets.QAction('&Save json', self)
        save_act.setStatusTip('Save as a DataPattern.json file')
        save_act.triggered.connect(self.save_dp_call)
        dp_menu.addAction(save_act)

        # Export as ascii pattern matrix
        exportascii_act = QtWidgets.QAction('&Export pattern', self)
        exportascii_act.setStatusTip('Export as an .txt .csv or .2db file')
        exportascii_act.triggered.connect(self.export_dp_call)
        dp_menu.addAction(exportascii_act)

        # Save as image
        saveimage_act = QtWidgets.QAction('&Save image', self)
        saveimage_act.setStatusTip('Save pattern as an image')
        saveimage_act.triggered.connect(self.saveasimage_dp_call)
        dp_menu.addAction(saveimage_act)

        # Copy to clipboard
        copy_act = QtWidgets.QAction('&Copy to clipboard', self)
        copy_act.setStatusTip('Copy pattern image to clipboard')
        copy_act.triggered.connect(self.call_copy_to_clipboard)
        dp_menu.addAction(copy_act)

        # Separate io from manipulation
        dp_menu.addSeparator()

        # Remove edge pixels
        removeedge_act = QtWidgets.QAction('&Remove edge pixels', self)
        removeedge_act.setStatusTip('Remove edge pixels from the pattern.')
        removeedge_act.triggered.connect(self.call_pb_removeedge)
        dp_menu.addAction(removeedge_act)

        return dp_menu

    def untoggle_other_pb(self, qtpushbutton):
        assert isinstance(qtpushbutton, QtWidgets.QPushButton)
        if qtpushbutton == self.pb_maskrectangle:
            self.untoggle_pb_maskpixel()
        elif qtpushbutton == self.pb_maskpixel:
            self.untoggle_pb_maskrectangle()
        else:
            self.untoggle_pb_maskpixel()
            self.untoggle_pb_maskrectangle()

    def untoggle_pb_maskrectangle(self):
        if self.pb_maskrectangle.isChecked():
            self.pb_maskrectangle.setChecked(False)
            if self.datapattern is not None:
                self.call_pb_maskrectangle(self.pb_maskrectangle)

    def untoggle_pb_maskpixel(self):
        if self.pb_maskpixel.isChecked():
            self.pb_maskpixel.setChecked(False)
            if self.datapattern is not None:
                self.call_pb_maskpixel(self.pb_maskpixel)

    #def set_datapattern(self, datapattern):
    #    self.dpcontroler.set_datapattern(datapattern)

    #def get_datapattern(self):
    #    return self.dpcontroler.get_datapattern()

    #def are_changes_saved(self):
    #    return self.dpcontroler.are_changes_saved()

    def set_dp_filename(self, filename: str):
        self.dp_filename = filename

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = DataPattern_window()
    window.show()
    print(window.size())
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
