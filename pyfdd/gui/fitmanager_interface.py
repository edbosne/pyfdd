
import sys
import os
import warnings
from enum import Enum, IntEnum
import numpy as np


from PyQt5 import QtCore, QtGui, QtWidgets, uic, sip
# from PySide2 import QtCore, QtGui, QtWidgets, uic

import pyfdd

# Load the ui created with PyQt creator
# First, convert .ui file to .py with,
# pyuic5 datapattern_widget.ui -o datapattern_widget.py
# import with absolute import locations
from pyfdd.gui.qt_designer.fitmanager_widget import Ui_FitManagerWidget
from pyfdd.gui.qt_designer.fitconfig_dialog import Ui_FitConfigDialog
from pyfdd.gui.qt_designer.parameteredit_dialog import Ui_ParameterEditDialog
from pyfdd.gui.viewresults_interface import ViewResults_widget
from pyfdd.gui.datapattern_interface import DataPattern_window


class Profile(IntEnum):
    coarse = 0
    default = 1
    fine = 2
    custom = 3


class CostFunc(IntEnum):
    chi2 = 0
    ml = 1


class FitConfig_dialog(QtWidgets.QDialog, Ui_FitConfigDialog):
    def __init__(self, parent_widget, current_config, fitman):
        assert isinstance(current_config, dict)
        assert isinstance(fitman, pyfdd.FitManager)
        super(FitConfig_dialog, self).__init__(parent_widget)
        self.setupUi(self)

        self.fitman = fitman
        self.load_config(current_config)
        self.new_config = dict

        # Connect signals
        self.cb_costfunc.currentIndexChanged.connect(self.update_profile_text)
        self.cb_profile.currentIndexChanged.connect(self.update_profile_text)

    def load_config(self, config):
        self.cb_costfunc.setCurrentIndex(config['cost_func'])
        self.ckb_geterrors.setChecked(config['get_errors'])
        self.sb_numsites.setValue(config['n_sites'])
        self.sb_subpixels.setValue(config['sub_pixels'])
        self.cb_profile.setCurrentIndex(int(config['min_profile']))
        if config['min_profile'] is Profile.custom:
            self.le_profile.setText(config['custom_profile'])
        else:
            self.update_profile_text()

    def update_profile_text(self):
        profile = Profile(self.cb_profile.currentIndex())
        cost_func = CostFunc(self.cb_costfunc.currentIndex())
        if profile is not Profile.custom:
            fit_options = self.fitman._profiles_fit_options[profile.name][cost_func.name]
            self.le_profile.setText(str(fit_options))
            self.le_profile.setReadOnly(True)
        else:
            self.le_profile.setReadOnly(False)

    def get_config(self):
        self.new_config = {
            'cost_func': CostFunc(self.cb_costfunc.currentIndex()),
            'get_errors': self.ckb_geterrors.isChecked(),
            'n_sites': self.sb_numsites.value(),
            'sub_pixels': self.sb_subpixels.value(),
            'min_profile': Profile(self.cb_profile.currentIndex())}

        if self.new_config['min_profile'] is Profile.custom:
            self.new_config['custom'] = self.le_profile.text()
        else:
            self.new_config['custom'] = ''

        return self.new_config


class ParameterEdit_dialog(QtWidgets.QDialog, Ui_ParameterEditDialog):
    def __init__(self, parent_widget, parameter):
        assert isinstance(parameter, Parameter)
        super(ParameterEdit_dialog, self).__init__(parent_widget)
        self.setupUi(self)

        self.le_initial_value.setText(str(parameter.initial_value))
        self.le_range_min.setText(str(parameter.bounds[0]))
        self.le_range_max.setText(str(parameter.bounds[1]))
        self.le_step_mod.setText(str(parameter.step_modifier))
        self.cb_fixed.setChecked(parameter.fixed)

    def get_initial_value(self):
        try:
            val = float(self.le_initial_value.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Initial value must be a float.')
        else:
            return val

    def get_bounds(self):
        try:
            text = self.le_range_min.text()
            val_min = None if text == 'None' else float(text)
            text = self.le_range_max.text()
            val_max = None if text == 'None' else float(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Range must be a float or None.')
        else:
            return [val_min, val_max]

    def get_step_modifier(self):
        try:
            val = float(self.le_step_mod.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Step modifier must be a float.')
        else:
            return val

    def get_fixed(self):
        return self.cb_fixed.isChecked()


class Parameter:
    def __init__(self, parent_widget, key='', name='par', initial_value=0, bounds=[None, None],
                 step_modifier=1, fixed=False, pb_edit=None, lb_name=None, lb_description=None):
        self.parent = parent_widget
        self.key = key
        self.name = name
        self.initial_value = initial_value
        self.bounds = bounds
        self.step_modifier = step_modifier
        self.fixed = fixed
        self.was_changed = False
        if pb_edit is not None:
            self.pb_edit = pb_edit
        else:
            self.pb_edit = QtWidgets.QPushButton(parent=parent_widget, text='Edit')
            self.pb_edit.setMaximumSize(QtCore.QSize(50, 16777215))

        if lb_name is not None:
            self.lb_name = lb_name
        else:
            self.lb_name = QtWidgets.QLabel(parent=parent_widget)
            self.lb_name.setText(name)
            # self.lb_name.setMaximumSize(QtCore.QSize(70, 16777215)) #double digit names need more space

        if lb_description is not None:
            self.lb_description = lb_description
        else:
            self.lb_description = QtWidgets.QLabel(parent=parent_widget)
            self.lb_description.setText('')
        self.update_description()

        # Connect signals
        self.pb_edit.clicked.connect(self.call_pb_edit)

    def reset_values_to(self, initial_value=0, bounds=[None, None], step_modifier=1, fixed=False):
        self.initial_value = initial_value
        self.bounds = bounds
        self.step_modifier = step_modifier
        self.fixed = fixed
        self.was_changed = False
        self.update_description()

    def update_description(self):
        # Print a '-' if there is no bound
        bounds = [a if a is not None else '-' for a in self.bounds]
        fixed = 'F' if self.fixed else ''
        if self.initial_value < 100:
            base_text = '{:.2f}; [{}, {}]; {:.2f}; {}'
        else:
            base_text = '{:.1e}; [{}, {}]; {:.2f}; {}'
        text = base_text.format(self.initial_value,
                                *bounds,
                                self.step_modifier,
                                fixed)
        self.lb_description.setText(text)

        if self.was_changed:
            self.lb_name.setStyleSheet("background-color:green;")
        else:
            # back to default
            self.lb_name.setStyleSheet('')

    def call_pb_edit(self):
        edit_dialog = ParameterEdit_dialog(parent_widget=self.parent, parameter=self)
        if edit_dialog.exec_() == QtWidgets.QDialog.Accepted:
            # only apply new value if it is not None
            value = edit_dialog.get_initial_value()
            self.initial_value = value if value is not None else self.initial_value
            value = edit_dialog.get_bounds()
            self.bounds = value if value is not None else self.bounds
            value = edit_dialog.get_step_modifier()
            self.step_modifier = value if value is not None else self.step_modifier
            self.fixed = edit_dialog.get_fixed()
            self.was_changed = True
            self.update_description()
        else:
            # Canceled
            pass

    def add_to_gridlayout(self, layout, row_num):
        assert isinstance(layout, QtWidgets.QGridLayout)
        layout.addWidget(self.lb_name, row_num, 0)
        layout.addWidget(self.lb_description, row_num, 1)
        layout.addWidget(self.pb_edit, row_num, 2)

    def __del__(self):
        """ Delete the parameter widgets once the last reference to the parameter instance is lost. """
        if not sip.isdeleted(self.lb_name):
            self.lb_name.deleteLater()
            self.lb_name = None
        if not sip.isdeleted(self.lb_description):
            self.lb_description.deleteLater()
            self.lb_description = None
        if not sip.isdeleted(self.pb_edit):
            self.pb_edit.deleteLater()
            self.pb_edit = None


class SiteRange:
    def __init__(self, parent_widget, key='', name='par', str_value='1',
                 lb_name=None, le_siterange=None):
        self.parent = parent_widget
        self.key = key
        self.name = name

        if lb_name is not None:
            self.lb_name = lb_name
        else:
            self.lb_name = QtWidgets.QLabel(parent=parent_widget)
            self.lb_name.setText(name)
            self.lb_name.setMaximumSize(QtCore.QSize(70, 16777215)) #double digit names need more space

        if le_siterange is not None:
            self.le_siterange = le_siterange
        else:
            self.le_siterange = QtWidgets.QLineEdit(parent=parent_widget)
            self.le_siterange.setText(str_value)

        # set regular expression validator
        reg_ex = QtCore.QRegExp(
            r'^(\s*\d+\s*(-\s*\d+\s*)?)(,\s*\d+\s*(-\s*\d+\s*)?)*$')  # accepts ranges ex.: 1,2,3, 6-9
        input_validator = QtGui.QRegExpValidator(reg_ex, parent=self.le_siterange)
        self.le_siterange.setValidator(input_validator)

    def add_to_gridlayout(self, layout, row_num):
        assert isinstance(layout, QtWidgets.QGridLayout)
        layout.addWidget(self.lb_name, row_num, 0)
        layout.addWidget(self.le_siterange, row_num, 1)

    def __del__(self):
        """ Delete the parameter widgets once the last reference to the parameter instance is lost. """
        if not sip.isdeleted(self.lb_name):
            self.lb_name.deleteLater()
            self.lb_name = None
        if not sip.isdeleted(self.le_siterange):
            self.le_siterange.deleteLater()
            self.le_siterange = None

    def get_range_as_list(self):
        result = []
        text = self.le_siterange.text()
        for part in text.split(','):
            if '-' in part:
                a, b = part.split('-')
                a, b = int(a), int(b)
                result.extend(range(a, b + 1))
            else:
                a = int(part)
                result.append(a)
        return result


class FitManawerWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress_msg = QtCore.pyqtSignal(str)
    output_fitman = QtCore.pyqtSignal(pyfdd.FitManager)

    def __init__(self, datapattern, simlibrary, fitconfig, parameter_objects, sites_range_objects):
        super(FitManawerWorker, self).__init__()

        self._isRunning = True

        # Define variables for creating a fitman
        cost_function = fitconfig['cost_func'].name
        n_sites = fitconfig['n_sites']
        sub_pixels = fitconfig['sub_pixels']
        self.get_errors = fitconfig['get_errors']

        # Create a fit manager
        self.fitman = pyfdd.FitManager(cost_function=cost_function,
                                       n_sites=n_sites,
                                       sub_pixels=sub_pixels)

        # Replace fitman print function
        self.fitman.print = self.new_print


        # Set the pattern and library to fit with
        self.fitman.set_pattern(datapattern, simlibrary)

        parameter_keys = self.fitman.parameter_keys

        # Verify all keys match
        for key, parameter in zip(parameter_keys, parameter_objects):
            assert parameter.key == key

        # Set a fixed value if needed
        fixed_values = {}
        for key, parameter in zip(parameter_keys, parameter_objects):
            if parameter.fixed:
                fixed_values[key] = parameter.initial_value
        self.fitman.set_fixed_values(**fixed_values)

        # Change default bounds
        bounds = {key: parameter.bounds
                  for key, parameter in zip(parameter_keys, parameter_objects)}
        self.fitman.set_bounds(**bounds)

        # Change default step modifier
        step_modifier = {key: parameter.step_modifier
                         for key, parameter in zip(parameter_keys, parameter_objects)}
        self.fitman.set_step_modifier(**step_modifier)

        # Change initial values
        initial_values = {key: parameter.initial_value
                          for key, parameter in zip(parameter_keys, parameter_objects)}
        self.fitman.set_initial_values(**initial_values)

        # Set a minization profile
        if fitconfig['min_profile'] is Profile.custom:
            min_profile = fitconfig['custom_profile']
        else:
            min_profile = fitconfig['min_profile'].name

        self.fitman.set_minimization_settings(profile=min_profile)

        # Set a pattern or range of patterns to fit
        self.sites_to_fit = [site_range.get_range_as_list() for site_range in sites_range_objects]

    @QtCore.pyqtSlot()
    def run(self):
        self.new_print('*'*80)
        self.new_print(' '*30 + 'PyFDD version {}'.format(pyfdd.__version__))
        self.new_print('*' * 80)

        # Run fits
        # remember to set get_errors to True if you want them. This increases the fit time.
        self.fitman.run_fits(*self.sites_to_fit, get_errors=self.get_errors)

        # Emit fitman for output
        self.output_fitman.emit(self.fitman)

        # Add a few blank lines for aestetics
        self.new_print('\n\n\n')

        # Finish
        self.finished.emit()

    @QtCore.pyqtSlot()
    def stop(self):
        self._isRunning = False
        self.fitman.stop_current_fit()

    def new_print(self, *msg):
        msg_str = [str(s) for s in msg]
        message = ' '.join(msg_str)
        # message = message + '\n'
        print(message)
        self.progress_msg.emit(message)


class FitManager_window(QtWidgets.QMainWindow):
    """ Class to use the data pattern widget in a separate window"""
    def __init__(self, *args, **kwargs):
        super(FitManager_window, self).__init__(*args, **kwargs)

        # Setup the window
        self.setWindowTitle("Fit Manager")
        self.statusBar()

        # Set a DataPattern widget as central widget
        dp_w = FitManager_widget(mainwindow=self)
        self.setCentralWidget(dp_w)
        self.resize(1150, 670)

    @staticmethod
    def get_datapattern():
        datapattern = pyfdd.DataPattern(
            '/home/eric/cernbox/PyCharm/PyFDD/test_pyfdd/test_files/pad_dp_2M.json')
        return datapattern

    @staticmethod
    def get_simlibrary():
        simlibrary = pyfdd.Lib2dl(
            '/home/eric/cernbox/PyCharm/PyFDD/test_pyfdd/test_files/sb600g05.2dl')
        return simlibrary


class FitManager_widget(QtWidgets.QWidget, Ui_FitManagerWidget):
    """ Data pattern widget class"""

    def __init__(self, *args, mainwindow=None, **kwargs):
        """
        Init method for the data pattern widget
        :param args:
        :param mainwindow: Main window object
        :param kwargs:
        """

        super(FitManager_widget, self).__init__(*args, **kwargs)

        # Alternative way to load the ui created with PyQt creator
        # uic.loadUi('qt_designer/datapattern_widget.ui', self)

        self.setupUi(self)
        self.mainwindow = mainwindow

        # Configure text browser font
        self.tb_fit_report.setFontFamily("monospace")

        # Create a menubar entry for the datapattern
        self.menubar = self.mainwindow.menuBar()
        self.dp_menu = self.setup_menu()

        # Dinamic widgets
        self.n_sites_widget_stack = []
        self.fractions_widget_stack = []
        self.n_sites_in_stack = 1

        # Popup widgets
        self.viewresults_window = None
        self.dp_external = []

        # Variables
        self.tr_costfunc = {'chi2': 'Chi-square',
                            'ml': 'Neg. log likelihood'}
        self.datapattern = None
        self.simlibrary = None
        self.get_datapattern()
        self.get_simlibrary()
        self.fitman = None

        # Fitman thread variables
        self.fitman_thread = None
        self.fitman_worker = None
        self.fitman_output = None

        # Fit configuration
        self.fitconfig = {'cost_func': CostFunc.chi2,
                          'get_errors': True,
                          'n_sites': 1,
                          'sub_pixels': 1,
                          'min_profile': Profile.default,
                          'custom_profile': ''}
        # Create a dummy fitmanager to correctly get the default fit parameters
        self.update_fitman()

        # Sites ranges
        self.sites_range_objects = []
        self.init_sites_ranges()

        # Parameters
        # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        self.parameter_objects = []
        self.parameter_keys = ()
        self.initial_values = dict()
        self.bounds = dict()
        self.step_modifier = dict()
        self.fixed = dict()
        self.init_parameters()
        self.refresh_parameters()

        # Connect signals
        self.pb_fitconfig.clicked.connect(self.call_pb_fitconfig)
        self.pb_reset.clicked.connect(self.reset_parameters)
        self.pb_abortfits.setEnabled(False)
        self.pb_runfits.clicked.connect(self.call_pb_runfits)
        self.pb_abortfits.clicked.connect(self.call_pb_abortfits)
        self.pb_viewresults.clicked.connect(self.call_pb_viewresults)
        self.pb_savetable.clicked.connect(self.call_pb_savetable)
        self.pb_viewfit.clicked.connect(self.call_pb_viewlastfit)
        self.pb_viewfitdiff.clicked.connect(self.call_pb_viewfitdiff)
        self.pb_filldata.clicked.connect(self.call_pb_filldata)

        self.update_infotext()

    def setup_menu(self):
        # nothing to do here at the moment
        return None

    def make_dummy_pattern(self):
        pattern = np.random.poisson(1000, (22, 22))
        dp = pyfdd.DataPattern(pattern_array=pattern)
        dp.manip_create_mesh(pixel_size=1.3, distance=300)
        return dp

    def init_parameters(self):
        # Parameters
        # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        # dx
        par = Parameter(parent_widget=self, key='dx', name='dx', initial_value=0, bounds=[None, None],
                        step_modifier=1, fixed=False, pb_edit=self.pb_dx, lb_name=self.lb_dx_name,
                        lb_description=self.lb_dx)
        self.parameter_objects.append(par)
        # dy
        par = Parameter(parent_widget=self, key='dy', name='dy', initial_value=0, bounds=[None, None],
                        step_modifier=1, fixed=False, pb_edit=self.pb_dy, lb_name=self.lb_dy_name,
                        lb_description=self.lb_dy)
        self.parameter_objects.append(par)
        # phi
        par = Parameter(parent_widget=self, key='phi', name='phi', initial_value=0, bounds=[None, None],
                        step_modifier=1, fixed=False, pb_edit=self.pb_phi, lb_name=self.lb_phi_name,
                        lb_description=self.lb_phi)
        self.parameter_objects.append(par)
        # total_cts
        par = Parameter(parent_widget=self, key='total_cts', name='total cts', initial_value=0, bounds=[None, None],
                        step_modifier=1, fixed=False, pb_edit=self.pb_total_cts, lb_name=self.lb_total_cts_name,
                        lb_description=self.lb_total_cts)
        self.parameter_objects.append(par)
        # sigma
        par = Parameter(parent_widget=self, key='sigma', name='sigma', initial_value=0, bounds=[None, None],
                        step_modifier=1, fixed=False, pb_edit=self.pb_sigma, lb_name=self.lb_sigma_name,
                        lb_description=self.lb_sigma)
        self.parameter_objects.append(par)
        # f_p1
        par = Parameter(parent_widget=self, key='f_p1', name='fraction #1', initial_value=0, bounds=[None, None],
                        step_modifier=1, fixed=False, pb_edit=self.pb_f1, lb_name=self.lb_f1_name,
                        lb_description=self.lb_f1)
        self.parameter_objects.append(par)

    def refresh_parameters(self, reset=False):
        self.parameter_keys = self.fitman.parameter_keys
        temp_init_values, temp_fixed_values = self.fitman._get_initial_values()  # fitman needs a dp to get the ini v
        self.initial_values = dict()
        self.fixed = dict()
        for key, value in zip(self.parameter_keys, temp_init_values):
            self.initial_values[key] = value
        for key, value in zip(self.parameter_keys, temp_fixed_values):
            self.fixed[key] = value
        self.bounds = self.fitman._bounds
        self.step_modifier = self.fitman._scale

        for key, parameter in zip(self.parameter_keys, self.parameter_objects):
            assert parameter.key == key
            if not reset and parameter.was_changed:
                # keep the value introduced by the user
                continue
            parameter.reset_values_to(initial_value=self.initial_values[key],
                                      bounds=self.bounds[key],
                                      step_modifier=self.step_modifier[key],
                                      fixed=self.fixed[key])

    def reset_parameters(self):
        self.refresh_parameters(reset=True)

    def init_sites_ranges(self):
        """ Create the first site range widget"""
        srange = SiteRange(parent_widget=self, key='sr1', name='Site #1',
                           lb_name=self.lb_f1_name,
                           le_siterange=self.le_site1)
        self.sites_range_objects.append(srange)

    def update_infotext(self):

        base_text = 'Data pattern set: {}; Library set: {}\n' \
                    'Cost function: {}; Get errors: {}\n' \
                    'Number of sites: {}; Sub-pixels: {}\n' \
                    'Miniminazion profile: {}'

        dp_set = False if self.datapattern is None else True
        lib_set = False if self.simlibrary is None else True
        cost_func = self.tr_costfunc[self.fitconfig['cost_func'].name]
        get_errors = self.fitconfig['get_errors']
        n_sites = self.fitconfig['n_sites']
        sub_pixels = self.fitconfig['sub_pixels']
        min_profile = self.fitconfig['min_profile'].name

        text = base_text.format(dp_set, lib_set, cost_func,
                                get_errors, n_sites,
                                sub_pixels,min_profile)

        self.infotext.setText(text)

    def get_datapattern(self):

        if self.mainwindow is None:
            self.datapattern = None
        else:
            self.datapattern = self.mainwindow.get_datapattern()

    def get_simlibrary(self):

        if self.mainwindow is None:
            self.simlibrary = None
        else:
            self.simlibrary = self.mainwindow.get_simlibrary()

    def call_pb_fitconfig(self):

        fitconfig_dialog = FitConfig_dialog(parent_widget=self, current_config=self.fitconfig, fitman=self.fitman)
        if fitconfig_dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.fitconfig = fitconfig_dialog.get_config()
            self.update_infotext()
            self.update_fitman()
            self.update_n_sites_widgets()
        else:
            # Canceled
            pass

    def call_pb_runfits(self):

        if self.datapattern is None:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Data pattern is not set.')
            return

        if self.simlibrary is None:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Simulation library is not set.')
            return

        # Clear text box
        self.tb_fit_report.clear()

        # Create a QThread object
        self.fitman_thread = QtCore.QThread()

        # Step 3: Create a worker object
        self.fitman_worker = FitManawerWorker(self.datapattern, self.simlibrary,
                                         self.fitconfig, self.parameter_objects,
                                         self.sites_range_objects)

        # Move worker to the thread
        self.fitman_worker.moveToThread(self.fitman_thread)

        # Connect signals and slots
        self.fitman_thread.started.connect(self.fitman_worker.run)
        self.fitman_worker.finished.connect(self.fitman_thread.quit)
        self.fitman_worker.finished.connect(self.fitman_worker.deleteLater)
        self.fitman_thread.finished.connect(self.fitman_thread.deleteLater)
        self.fitman_worker.progress_msg.connect(self.report_fit_progress)
        # output signal
        self.fitman_worker.output_fitman.connect(self.call_store_output)

        # Manage buttons
        # Activate - Deactivate abort button
        self.pb_abortfits.setEnabled(True)
        self.fitman_thread.finished.connect(
            lambda: self.pb_abortfits.setEnabled(False)
        )
        # Deactivate - Activate runfits button
        self.pb_runfits.setEnabled(False)
        self.fitman_thread.finished.connect(
            lambda: self.pb_runfits.setEnabled(True)
        )

        # Start the thread
        self.fitman_thread.start()

    def report_fit_progress(self, msg):
        self.tb_fit_report.append(msg)

    def call_pb_abortfits(self):

        resp = QtWidgets.QMessageBox.question(self, 'Warning', "Do you want to abort?",
                                              QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if resp == QtWidgets.QMessageBox.Yes:
            if self.fitman_worker is not None:
                self.fitman_worker.stop()

    def call_store_output(self, fitman):
        self.fitman_output = fitman

    def call_pb_viewresults(self):

        if self.fitman_output is not None:
            self.viewresults_window = ViewResults_widget(results_df=self.fitman_output.df_horizontal,
                                                    parent_widget=self)
            self.viewresults_window.show()
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Results are not ready.')

    def call_pb_savetable(self):

        if self.fitman_output is not None:
            filename = QtWidgets.QFileDialog. \
                getSaveFileName(self, 'Export DataPattern',
                                filter='data (*csv *txt)',
                                options=QtWidgets.QFileDialog.DontUseNativeDialog)

            if filename == ('', ''):  # Cancel
                return

            self.fitman_output.save_output(filename[0])
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Results are not ready.')

    def call_pb_viewlastfit(self):

        if self.fitman_output is not None:
            items = ("counts", "yield", "probability")
            normalization, ok = QtWidgets.QInputDialog.getItem(self, "Choose normalization",
                                                   "Normalization:", items, 0, False)
            if ok and normalization:
                datapattern = self.fitman_output.get_pattern_from_last_fit( normalization=normalization)
                new_dp_window = DataPattern_window()
                new_dp_window.set_datapattern(datapattern)
                new_dp_window.setWindowTitle('Last Fit Pattern')
                new_dp_window.show()
                self.dp_external.append(new_dp_window)
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Results are not ready.')

    def call_pb_viewfitdiff(self):

        if self.fitman_output is not None:
            items = ("counts", "yield", "probability")
            normalization, ok = QtWidgets.QInputDialog.getItem(self, "Choose normalization",
                                                      "Normalization:", items, 0, False)
            if ok and normalization:
                datapattern_data = self.fitman_output.get_datapattern(normalization=normalization)
                datapattern_fit = self.fitman_output.get_pattern_from_last_fit(normalization=normalization)
                new_dp_window = DataPattern_window()
                new_dp_window.set_datapattern(datapattern_data - datapattern_fit)
                new_dp_window.setWindowTitle('Data Pattern - Fit Pattern')
                new_dp_window.show()
                self.dp_external.append(new_dp_window)
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Results are not ready.')

    def call_pb_filldata(self):

        if self.fitman_output is not None:
            items = ("counts", "yield", "probability")
            normalization, ok = QtWidgets.QInputDialog.getItem(self, "Choose normalization",
                                                      "Normalization:", items, 0, False)
            if ok and normalization:
                items = ('ideal', 'poisson', 'montecarlo')
                generator, ok = QtWidgets.QInputDialog.getItem(self, "Choose generator",
                                                                   "Generator:", items, 0, False)
                if ok and generator:
                    datapattern = self.fitman_output.get_datapattern(normalization=normalization,
                                                                     substitute_masked_with=generator)
                    new_dp_window = DataPattern_window()
                    new_dp_window.set_datapattern(datapattern)
                    new_dp_window.setWindowTitle('Data Pattern with Replaced Masked Pixels')
                    new_dp_window.show()
                    self.dp_external.append(new_dp_window)
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Results are not ready.')

    def update_fitman(self):
        cost_function = self.fitconfig['cost_func'].name
        n_sites = self.fitconfig['n_sites']
        sub_pixels = self.fitconfig['sub_pixels']
        self.fitman = pyfdd.FitManager(cost_function=cost_function,
                                       n_sites=n_sites,
                                       sub_pixels=sub_pixels)
        if self.datapattern is not None:
            self.fitman.dp_pattern = self.datapattern
        else:
            self.fitman.dp_pattern = self.make_dummy_pattern()
        if self.simlibrary is not None:
            self.fitman.lib = self.simlibrary

    def update_all(self):
        self.get_datapattern()
        self.get_simlibrary()
        self.update_fitman()
        self.update_infotext()

    def update_n_sites_widgets(self):
        self.parameters_layout.removeWidget(self.pb_reset)
        if self.fitconfig['n_sites'] > self.n_sites_in_stack:
            # add widgets
            while self.fitconfig['n_sites'] > self.n_sites_in_stack:
                self.n_sites_in_stack += 1
                # Sites ranges
                srkey = 'sr' + str(self.n_sites_in_stack)
                # srx
                site_name = 'Site #{}'.format(self.n_sites_in_stack)
                srange = SiteRange(parent_widget=self, key=srkey, name=site_name)
                srange.add_to_gridlayout(self.sitesrange_layout, row_num=1 + self.n_sites_in_stack)
                self.sites_range_objects.append(srange)

                # Parameters
                pkey = 'f_p' + str(self.n_sites_in_stack)
                # f_px
                fraction_name = 'fraction #{}'.format(self.n_sites_in_stack)
                par = Parameter(parent_widget=self, key=pkey, name=fraction_name, initial_value=0,
                                bounds=[None, None],
                                step_modifier=1, fixed=False)
                par.add_to_gridlayout(self.parameters_layout, row_num=5 + self.n_sites_in_stack)
                self.parameter_objects.append(par)
            self.refresh_parameters()

        if self.fitconfig['n_sites'] < self.n_sites_in_stack:
            while self.fitconfig['n_sites'] < self.n_sites_in_stack:
                self.n_sites_in_stack -= 1
                self.sites_range_objects.pop()
                self.parameter_objects.pop()

        self.parameters_layout.addWidget(self.pb_reset, 5 + 1 + self.n_sites_in_stack, 2)


def main():
    app = QtWidgets.QApplication(sys.argv)
    # window = DataPattern_widget()
    window = FitManager_window()
    window.show()
    print(window.size())
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
