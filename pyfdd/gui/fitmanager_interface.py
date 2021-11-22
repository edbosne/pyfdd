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
from pyfdd.gui.qt_designer.fitmanager_widget import Ui_FitManagerWidget
from pyfdd.gui.qt_designer.fitconfig_dialog import Ui_FitConfigDialog
from pyfdd.gui.qt_designer.parameteredit_dialog import Ui_ParameterEditDialog
from pyfdd.gui.viewresults_interface import ViewResults_widget
from pyfdd.gui.datapattern_interface import DataPattern_window
import pyfdd.gui.config as config


class Profile(IntEnum):
    coarse = 0
    default = 1
    fine = 2
    custom = 3


class CostFunc(IntEnum):
    chi2 = 0
    ml = 1


class FitConfig_dialog(QtWidgets.QDialog, Ui_FitConfigDialog):
    def __init__(self, parent_widget, current_config):
        assert isinstance(current_config, dict)
        super(FitConfig_dialog, self).__init__(parent_widget)
        self.setupUi(self)

        self.custom_profile_str = None
        self.load_config(current_config)
        self.new_config = dict()

        # Connect signals
        self.cb_costfunc.currentIndexChanged.connect(self.update_profile_text)
        self.cb_profile.currentIndexChanged.connect(self.update_profile_text)
        self.le_profile.editingFinished.connect(self.custom_profile_changed)

    def load_config(self, config):
        self.cb_costfunc.setCurrentIndex(config['cost_func'])
        self.ckb_geterrors.setChecked(config['get_errors'])
        self.sb_numsites.setValue(config['n_sites'])
        self.sb_subpixels.setValue(config['sub_pixels'])
        self.cb_profile.setCurrentIndex(int(config['min_profile']))
        if config['min_profile'] is Profile.custom:
            self.le_profile.setText(str(config['custom_profile']))
            self.custom_profile_str = str(config['custom_profile'])
        else:
            self.update_profile_text()

    def update_profile_text(self):
        profile = Profile(self.cb_profile.currentIndex())
        cost_func = CostFunc(self.cb_costfunc.currentIndex())
        if profile is not Profile.custom:
            fit_options = pyfdd.FitManager.default_profiles_fit_options.copy()[profile.name][cost_func.name]
            self.le_profile.setText(str(fit_options))
            self.le_profile.setReadOnly(True)
        else:
            if self.custom_profile_str is not None:
                self.le_profile.setText(self.custom_profile_str)
            self.le_profile.setReadOnly(False)

    def custom_profile_changed(self):
        profile = Profile(self.cb_profile.currentIndex())
        if profile is Profile.custom:
            self.custom_profile_str = self.le_profile.text()

    def get_config(self):
        self.new_config = {
            'cost_func': CostFunc(self.cb_costfunc.currentIndex()),
            'get_errors': self.ckb_geterrors.isChecked(),
            'n_sites': self.sb_numsites.value(),
            'sub_pixels': self.sb_subpixels.value(),
            'min_profile': Profile(self.cb_profile.currentIndex())}

        if self.new_config['min_profile'] is Profile.custom:
            temp_string = self.le_profile.text()
            temp_string = temp_string.replace('\'', '\"')  # Use double quotes
            temp_string = temp_string.replace('False', 'false')  # Use lower case bools for json
            temp_string = temp_string.replace('True', 'true')
            print(temp_string)
            self.new_config['custom_profile'] = json.loads(temp_string)
            print(type(self.new_config['custom_profile']))
            print(self.new_config['custom_profile'])
        else:
            self.new_config['custom_profile'] = ''

        return self.new_config


class ParameterEdit_dialog(QtWidgets.QDialog, Ui_ParameterEditDialog):
    def __init__(self, parent_widget, parameter):
        assert isinstance(parameter, FitParameter)
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
            return val_min, val_max

    def get_step_modifier(self):
        try:
            val = float(self.le_step_mod.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Step modifier must be a float.')
        else:
            return val

    def get_fixed(self):
        return self.cb_fixed.isChecked()


class FitParameter:
    def __init__(self, parent_widget, key='', name='par', initial_value=0, bounds=(None, None),
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

    def reset_values_to(self, initial_value=0, bounds=(None, None), step_modifier=1, fixed=False):
        self.initial_value = initial_value
        self.bounds = bounds
        self.step_modifier = step_modifier
        self.fixed = fixed
        self.was_changed = False
        self.update_description()

    def update_description(self):
        # Print a '-' if there is no bound
        bounds = tuple([a if a is not None else '-' for a in self.bounds])
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
            self.lb_name.setStyleSheet("background-color:gray;")
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
    def __init__(self, parent_widget, key='', name='par', str_value='1', multiple_sites=True,
                 lb_name=None, le_siterange=None):
        self.parent = parent_widget
        self.key = key
        self.name = name

        if lb_name is not None:
            self.lb_name = lb_name
        else:
            self.lb_name = QtWidgets.QLabel(parent=parent_widget)
            self.lb_name.setText(name)
            self.lb_name.setMaximumSize(QtCore.QSize(70, 16777215))  # double digit names need more space

        if le_siterange is not None:
            self.le_siterange = le_siterange
        else:
            self.le_siterange = QtWidgets.QLineEdit(parent=parent_widget)
            self.le_siterange.setText(str_value)

        # set regular expression validator
        if multiple_sites:
            # accepts ranges ex.: 1,2,3, 6-9
            reg_ex = QtCore.QRegExp(
                r'^(\s*\d+\s*(-\s*\d+\s*)?)(,\s*\d+\s*(-\s*\d+\s*)?)*$')
        else:
            # accepts single numbers ex.: 1,2,3
            reg_ex = QtCore.QRegExp(
                r'^(\s*\d+\s*)?$')

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
        if len(text) == 0:
            QtWidgets.QMessageBox.warning(self.parent, 'Warning message', 'Empty site range.')
            raise ValueError('Site range is empty.')

        for part in text.split(','):
            if '-' in part:
                a, b = part.split('-')
                a, b = int(a), int(b)
                result.extend(range(a, b + 1))
            else:
                a = int(part)
                result.append(a)
        return result


class FitParameterDynamicLayout:
    def __init__(self, parent_widget, grid_layout, exclude=None, fitparameter_type=FitParameter):

        # Variables
        self.parent = parent_widget
        self.parameters_layout = grid_layout
        self.fitparameter_type = fitparameter_type
        self.exclude = list() if exclude is None else exclude
        self.base_par = 0

        # Parameters
        # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        self.n_sites_in_stack = 1
        self.parameter_objects = []
        self.init_parameters()
        self.refresh_parameters(reset=True)

    def init_parameters(self):
        # Parameters
        # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        # dx
        if 'dx' not in self.exclude:
            par = self.fitparameter_type(parent_widget=self.parent, key='dx', name='dx', initial_value=0,
                               bounds=[None, None], step_modifier=1, fixed=False,
                               pb_edit=self.parent.pb_dx,
                               lb_name=self.parent.lb_dx_name,
                               lb_description=self.parent.lb_dx)
            self.parameter_objects.append(par)
            self.base_par += 1
        # dy
        if 'dy' not in self.exclude:
            par = self.fitparameter_type(parent_widget=self.parent, key='dy', name='dy', initial_value=0,
                               bounds=[None, None], step_modifier=1, fixed=False,
                               pb_edit=self.parent.pb_dy,
                               lb_name=self.parent.lb_dy_name,
                               lb_description=self.parent.lb_dy)
            self.parameter_objects.append(par)
            self.base_par += 1
        # phi
        if 'phi' not in self.exclude:
            par = self.fitparameter_type(parent_widget=self.parent, key='phi', name='phi', initial_value=0,
                                         bounds=[None, None], step_modifier=1, fixed=False,
                                         pb_edit=self.parent.pb_phi,
                                         lb_name=self.parent.lb_phi_name,
                                         lb_description=self.parent.lb_phi)
            self.parameter_objects.append(par)
            self.base_par += 1
        # total_cts
        if 'total_cts' not in self.exclude:
            par = self.fitparameter_type(parent_widget=self.parent, key='total_cts', name='total cts', initial_value=0,
                                         bounds=[None, None], step_modifier=1, fixed=False,
                                         pb_edit=self.parent.pb_total_cts,
                                         lb_name=self.parent.lb_total_cts_name,
                                         lb_description=self.parent.lb_total_cts)
            self.parameter_objects.append(par)
            self.base_par += 1
        # sigma
        if 'sigma' not in self.exclude:
            par = self.fitparameter_type(parent_widget=self.parent, key='sigma', name='sigma', initial_value=0,
                                         bounds=[None, None], step_modifier=1, fixed=False,
                                         pb_edit=self.parent.pb_sigma,
                                         lb_name=self.parent.lb_sigma_name,
                                         lb_description=self.parent.lb_sigma)
            self.parameter_objects.append(par)
            self.base_par += 1
        # f_p1
        if 'f_p1' not in self.exclude:
            par = self.fitparameter_type(parent_widget=self.parent, key='f_p1', name='fraction #1', initial_value=0,
                                         bounds=[None, None], step_modifier=1, fixed=False,
                                         pb_edit=self.parent.pb_f1,
                                         lb_name=self.parent.lb_f1_name,
                                         lb_description=self.parent.lb_f1)
            self.parameter_objects.append(par)

    def refresh_parameters(self, datapattern: pyfdd.DataPattern = None, reset: bool = False):
        """
        Refresh the parameters acoording to the current data pattern and library.
        :param reset: If true all paremeters that were changed by the user are reset.
        :param datapattern: DataPattern to use for initial values.
        :return:
        """

        if isinstance(datapattern, pyfdd.DataPattern):
            data_p = datapattern
        else:
            data_p = self.make_dummy_pattern()

        # Compute values
        fitparameters = pyfdd.FitParameters(n_sites=self.n_sites_in_stack)
        fitparameters.update_initial_values_with_datapattern(datapattern=data_p)
        fitparameters.update_bounds_with_datapattern(datapattern=data_p)

        parameter_keys = fitparameters.get_keys()
        for rmkey in self.exclude:  # remove keys in exclude
            if rmkey in parameter_keys:
                parameter_keys.remove(rmkey)
        initial_values = fitparameters.get_initial_values()
        fixed_values = fitparameters.get_fixed_values()
        bounds = fitparameters.get_bounds()
        step_modifier = fitparameters.get_step_modifier()

        # Apply new values to the interface
        for key, parameter in zip(parameter_keys, self.parameter_objects):
            assert parameter.key == key
            if not reset and parameter.was_changed:
                # Keep the value introduced by the user
                continue
            else:
                parameter.reset_values_to(initial_value=initial_values[key],
                                          bounds=bounds[key],
                                          step_modifier=step_modifier[key],
                                          fixed=fixed_values[key])

    def update_n_sites_widgets(self, n_sites):
        if 'reset' not in self.exclude:
            self.parameters_layout.removeWidget(self.parent.pb_reset)

        if n_sites > self.n_sites_in_stack:
            # add widgets
            while n_sites > self.n_sites_in_stack:
                self.n_sites_in_stack += 1

                # Parameters
                pkey = 'f_p' + str(self.n_sites_in_stack)
                # f_px
                fraction_name = 'fraction #{}'.format(self.n_sites_in_stack)
                par = self.fitparameter_type(parent_widget=self.parent, key=pkey, name=fraction_name, initial_value=0,
                                   bounds=(None, None),
                                   step_modifier=1, fixed=False)
                par.add_to_gridlayout(self.parameters_layout, row_num=self.base_par + self.n_sites_in_stack)
                self.parameter_objects.append(par)
            self.refresh_parameters()

        if n_sites < self.n_sites_in_stack:
            while n_sites < self.n_sites_in_stack:
                self.n_sites_in_stack -= 1
                self.parameter_objects.pop()

        if 'reset' not in self.exclude:
            self.parameters_layout.addWidget(self.parent.pb_reset, self.base_par + 1 + self.n_sites_in_stack, 2)

    @staticmethod
    def make_dummy_pattern():
        pattern = np.random.poisson(1000, (22, 22))
        dp = pyfdd.DataPattern(pattern_array=pattern)
        dp.manip_create_mesh(pixel_size=1.4, distance=300)
        return dp

    def get_parameter_keys(self):
        parameter_keys = []
        for parameter in self.parameter_objects:
            parameter_keys.append(parameter.key)
        return parameter_keys

    def get_fixed_values(self):
        parameter_keys = self.get_parameter_keys()

        fixed_values = dict()
        for key, parameter in zip(parameter_keys, self.parameter_objects):
            if parameter.fixed:
                fixed_values[key] = parameter.initial_value
        return fixed_values

    def get_bounds(self):
        parameter_keys = self.get_parameter_keys()

        bounds = {key: parameter.bounds
                  for key, parameter in zip(parameter_keys, self.parameter_objects)}
        return bounds

    def get_step_modifier(self):
        parameter_keys = self.get_parameter_keys()

        step_modifier = {key: parameter.step_modifier
                         for key, parameter in zip(parameter_keys, self.parameter_objects)}
        return step_modifier

    def get_initial_values(self):
        parameter_keys = self.get_parameter_keys()

        # Change initial values
        initial_values = {key: parameter.initial_value
                          for key, parameter in zip(parameter_keys, self.parameter_objects)}
        return initial_values


class SiteRangeDynamicLayout:
    def __init__(self, parent_widget, grid_layout, lb_f1_name, le_site1, multiple_sites=True):

        assert isinstance(lb_f1_name, QtWidgets.QWidget)
        assert isinstance(le_site1, QtWidgets.QWidget)

        # Variables
        self.multiple_sites = multiple_sites
        self.parent_widget = parent_widget

        # Expected existing widgets
        self.lb_f1_name = lb_f1_name
        self.le_site1 = le_site1
        self.sitesrange_layout = grid_layout

        # Sites ranges
        self.n_sites_in_stack = 1
        self.sites_range_objects = []
        self.init_sites_ranges()

    def init_sites_ranges(self):
        """ Create the first site range widget"""
        srange = SiteRange(parent_widget=self, key='sr1', name='Site #1', multiple_sites=self.multiple_sites,
                           lb_name=self.lb_f1_name,
                           le_siterange=self.le_site1)
        self.sites_range_objects.append(srange)

    def update_n_sites_widgets(self, n_sites):
        if n_sites > self.n_sites_in_stack:
            # add widgets
            while n_sites > self.n_sites_in_stack:
                self.n_sites_in_stack += 1
                # Sites ranges
                srkey = 'sr' + str(self.n_sites_in_stack)
                # srx
                site_name = 'Site #{}'.format(self.n_sites_in_stack)
                srange = SiteRange(parent_widget=self.parent_widget, key=srkey, name=site_name,
                                   multiple_sites=self.multiple_sites)
                srange.add_to_gridlayout(self.sitesrange_layout, row_num=1 + self.n_sites_in_stack)
                self.sites_range_objects.append(srange)

        if n_sites < self.n_sites_in_stack:
            while n_sites < self.n_sites_in_stack:
                self.n_sites_in_stack -= 1
                self.sites_range_objects.pop()

    def get_sites_for_fit(self):
        sites_to_fit = [site_range.get_range_as_list() for site_range in self.sites_range_objects]
        return sites_to_fit


class FitManawerWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress_msg = QtCore.pyqtSignal(str)
    output_fitman = QtCore.pyqtSignal(pyfdd.FitManager)

    def __init__(self, datapattern, simlibrary, fitconfig, dynamic_parameter_objects, dynamic_site_ranges_objects):
        super(FitManawerWorker, self).__init__()

        assert isinstance(dynamic_parameter_objects, FitParameterDynamicLayout)
        assert isinstance(dynamic_site_ranges_objects, SiteRangeDynamicLayout)

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
        self.fitman._print = self.new_print

        # Set the pattern and library to fit with
        self.fitman.set_pattern(datapattern, simlibrary)

        parameter_keys = self.fitman.fit_parameters.get_keys()

        # Verify all keys match
        assert parameter_keys == dynamic_parameter_objects.get_parameter_keys()

        # Set a fixed value if needed
        fixed_values = dynamic_parameter_objects.get_fixed_values()
        self.fitman.fit_parameters.change_fixed_values(**fixed_values)

        # Change default bounds
        bounds = dynamic_parameter_objects.get_bounds()
        self.fitman.fit_parameters.change_bounds(**bounds)

        # Change default step modifier
        step_modifier = dynamic_parameter_objects.get_step_modifier()
        self.fitman.fit_parameters.change_step_modifier(**step_modifier)

        # Change initial values
        initial_values = dynamic_parameter_objects.get_initial_values()
        self.fitman.fit_parameters.change_initial_values(**initial_values)

        # Set a minization profile
        if fitconfig['min_profile'] is Profile.custom:
            min_profile = 'custom'
            fit_options = fitconfig['custom_profile']
        else:
            min_profile = fitconfig['min_profile'].name
            fit_options = None

        self.fitman.set_minimization_settings(profile=min_profile, options=fit_options)

        # Set a pattern or range of patterns to fit
        self.sites_to_fit = dynamic_site_ranges_objects.get_sites_for_fit()

    def is_datapattern_inrange(self):
        return self.fitman.is_datapattern_inrange()

    @QtCore.pyqtSlot()
    def run(self):
        self.new_print('*' * 80)
        self.new_print(' ' * 30 + 'PyFDD version {}'.format(pyfdd.__version__))
        self.new_print('*' * 80)

        # Run fits
        # remember to set get_errors to True if you want them. This increases the fit time.
        print(f'self.sites_to_fit {self.sites_to_fit}')
        self.fitman.run_fits(*self.sites_to_fit, get_errors=self.get_errors)

        # Emit fitman for output
        self.output_fitman.emit(self.fitman)

        # Add a few blank lines for aestetics
        self.new_print('\n\nDone!\n\n\n')

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

        # Load configuration
        if config.parser is None:
            config.filename = 'fitmanager_config.ini'
            config.read()

        # Setup the window
        self.window_title = "Fit Manager"
        self.setWindowTitle(self.window_title)
        self.statusBar()

        # Set a DataPattern widget as central widget
        self.fm_w = FitManager_widget(mainwindow=self)
        self.setCentralWidget(self.fm_w)
        self.resize(1150, 670)

        # Connect
        self.fm_w.fitresults_changed.connect(self.title_update)
        self.fm_w.fitresults_saved.connect(self.title_update)

    @staticmethod
    def get_datapattern():
        datapattern = pyfdd.DataPattern(
            '../../test_pyfdd/data_files/pad_dp_2M.json')
        return datapattern

    @staticmethod
    def get_simlibrary():
        simlibrary = pyfdd.Lib2dl(
            '../../test_pyfdd/data_files/sb600g05.2dl')
        return simlibrary

    def title_update(self):
        if self.fm_w.are_changes_saved() is False:
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


class FitManager_widget(QtWidgets.QWidget, Ui_FitManagerWidget):
    """ Data pattern widget class"""

    fitresults_changed = QtCore.pyqtSignal()
    fitresults_saved = QtCore.pyqtSignal()

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
        # The Courier font is monospaced and works for both Windows and Linux
        self.setStyleSheet('QTextBrowser#tb_fit_report {font-family: Courier}')

        # Set config section
        if not config.parser.has_section('fitmanager'):
            config.parser.add_section('fitmanager')

        # Configure text browser font
        self.tb_fit_report.setFontFamily("monospace")

        # Create a menubar entry for the datapattern
        self.menubar = self.mainwindow.menuBar()
        self.dp_menu = self.setup_menu()

        # Popup widgets that need a reference in self
        self.viewresults_window = None  # todo multiple windows can be opened, but where are the references stored?
        self.dp_external = []

        # Variables
        self.tr_costfunc = {'chi2': 'Chi-square',
                            'ml': 'Neg. log likelihood'}
        self.datapattern = None
        self.simlibrary = None
        self.get_datapattern()
        self.get_simlibrary()
        self.changes_saved = True

        # Fitman thread variables
        self.fitman_thread = None
        self.fitman_worker = None
        self.fitman_output = None

        # Fit configuration
        default_fitconfig = {'cost_func': CostFunc.chi2,
                             'get_errors': True,
                             'n_sites': 1,
                             'sub_pixels': 1,
                             'min_profile': Profile.default,
                             'custom_profile': ''}

        if not config.parser.has_option('fitmanager', 'fitconfig'):
            self.fitconfig = default_fitconfig.copy()
        else:
            self.fitconfig = config.getdict('fitmanager', 'fitconfig')
            # convert ints to enum
            self.fitconfig['cost_func'] = CostFunc(self.fitconfig['cost_func'])
            self.fitconfig['min_profile'] = Profile(self.fitconfig['min_profile'])

        # Sites ranges
        self.dynamic_site_ranges = SiteRangeDynamicLayout(parent_widget=self,
                                                          grid_layout=self.sitesrange_layout,
                                                          lb_f1_name=self.lb_f1_name,
                                                          le_site1=self.le_site1,
                                                          multiple_sites=True)

        # Parameters
        # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        self.dynamic_parameters = FitParameterDynamicLayout(parent_widget=self,
                                                            grid_layout=self.parameters_layout,
                                                            exclude=None)

        # Connect signals
        self.pb_fitconfig.clicked.connect(self.call_pb_fitconfig)
        self.pb_reset.clicked.connect(lambda: self.dynamic_parameters.refresh_parameters(reset=True))
        self.pb_abortfits.setEnabled(False)
        self.pb_runfits.clicked.connect(self.call_pb_runfits)
        self.pb_abortfits.clicked.connect(self.call_pb_abortfits)
        self.pb_viewresults.clicked.connect(self.call_pb_viewresults)
        self.pb_savetable.clicked.connect(self.call_pb_savetable)
        self.pb_viewfit.clicked.connect(self.call_pb_viewlastfit)
        self.pb_viewfitdiff.clicked.connect(self.call_pb_viewfitdiff)
        self.pb_filldata.clicked.connect(self.call_pb_filldata)

        self.update_infotext()
        self.update_n_sites_widgets()

    def setup_menu(self):
        # nothing to do here at the moment
        return None

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
                                sub_pixels, min_profile)

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

    def are_changes_saved(self):
        return self.changes_saved

    def call_pb_fitconfig(self):

        fitconfig_dialog = FitConfig_dialog(parent_widget=self, current_config=self.fitconfig)
        if fitconfig_dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.fitconfig = fitconfig_dialog.get_config()
            self.update_infotext()
            self.update_n_sites_widgets()
            config.parser['fitmanager']['fitconfig'] = json.dumps(self.fitconfig)
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
                                              self.fitconfig, self.dynamic_parameters,
                                              self.dynamic_site_ranges)

        # Do checks before starting
        # Check if the fit range is correct
        if not self.fitman_worker.is_datapattern_inrange():
            QtWidgets.QMessageBox.warning(self, 'Warning message',
                                          'The datapattern is not in the simulation range. \n'
                                          'Consider reducing the fit range arount the axis first.')
            return

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
        self.changes_saved = False
        self.fitresults_changed.emit()

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
            self.changes_saved = True
            self.fitresults_saved.emit()
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning message', 'Results are not ready.')

    def call_pb_viewlastfit(self):

        if self.fitman_output is not None:
            items = ("counts", "yield", "probability")
            normalization, ok = QtWidgets.QInputDialog.getItem(self, "Choose normalization",
                                                               "Normalization:", items, 0, False)
            if ok and normalization:
                datapattern = self.fitman_output.get_pattern_from_last_fit(normalization=normalization)
                datapattern.clear_mask()
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

    def update_all(self):
        self.get_datapattern()
        self.get_simlibrary()
        self.update_infotext()
        self.refresh_parameters()

    def update_n_sites_widgets(self):
        self.dynamic_site_ranges.update_n_sites_widgets(self.fitconfig['n_sites'])
        self.dynamic_parameters.update_n_sites_widgets(self.fitconfig['n_sites'])


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FitManager_window()
    window.show()
    print(window.size())
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
