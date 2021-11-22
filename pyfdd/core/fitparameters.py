
"""
FitParameters is a class for holdind and editing fit parameters
"""

# Imports from standard library
import math
import copy

# Imports from 3rd party
import numpy as np

# Imports from project
from pyfdd.core.datapattern import DataPattern


class FitParameters:
    """
    Class to hold and manage fit parameters.
    """

    default_parameter_keys = ['dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_px']
    # total_cts is overwriten with values from the data pattern
    default_initial_values = {'dx': 0, 'dy': 0, 'phi': 0, 'total_cts': 1, 'sigma': 0, 'f_px': 0.15}
    default_fixed_values = {'dx': False, 'dy': False, 'phi': False, 'total_cts': True, 'sigma': True, 'f_px': False}
    default_scale = {'dx': .01, 'dy': .01, 'phi': 0.10, 'total_cts': 0.01,
                     'sigma': .001, 'f_px': 0.01}
    default_bounds = {'dx': (-3, +3), 'dy': (-3, +3), 'phi': (None, None), 'total_cts': (1, None),
                      'sigma': (0.01, None), 'f_px': (0, 1)}

    def __init__(self, n_sites):
        """
        Create a FitParameters object.
        :param n_sites: number of sites to use in the fit.
        :param fitman:
        """

        if not isinstance(n_sites, int):
            raise ValueError('n_sites should be an int.')
        self._n_sites = n_sites

        # Parameter keys
        self._parameter_keys = self._compute_keys()

        # Scale and Bounds
        self._scale = self._compute_step_modifier()
        self._bounds = self._compute_bounds()

        # overwrite defaults from Fit
        self._initial_values, self._fixed_values = self._compute_initial_values()

    def __str__(self):
        return_str = ''
        return_str += 'Parameter settings\n'
        return_str += '{:<16}{:<16}{:<16}{:<16}{:<16}\n'.format('Name', 'Initial Value', 'Fixed', 'Bounds', 'Scale')

        string_temp = '{:<16}{:<16.2f}{:<16}{:<16}{:<16}\n'
        for key in self._parameter_keys:
            # {'p0':None, 'value':None, 'fixed':False, 'std':None, 'scale':1, 'bounds':(None,None)}
            return_str += string_temp.format(
                                            key,
                                            self._initial_values[key],
                                            self._fixed_values[key],
                                            '({},{})'.format(*self._bounds[key]),
                                            self._scale[key]
                                            )
        return return_str

    def copy(self):
        return copy.deepcopy(self)

    def get_keys(self):
        return self._parameter_keys.copy()

    def get_bounds(self):
        return self._bounds.copy()

    def get_step_modifier(self):
        return self._scale.copy()

    def get_initial_values(self):
        return self._initial_values.copy()

    def get_fixed_values(self):
        return self._fixed_values.copy()

    def _compute_keys(self):
        """
        Compute the parameter keys for the number of sites configured at initialization.
        :return: List of parameter keys
        """
        # Set the parameter keys
        # Example: ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        parameter_keys = FitParameters.default_parameter_keys.copy()
        parameter_keys.pop()  # remove 'f_px'
        for i in range(self._n_sites):
            fraction_key = 'f_p' + str(i + 1)  # 'f_p1', 'f_p2', 'f_p3',...
            parameter_keys.append(fraction_key)

        return parameter_keys

    def _compute_initial_values(self):
        """
        Static method to get the defaut initial values for fitting.
        :return: p0, p_fix: initial values and tuple of bools indicating if it is fixed
        """

        initial_values = FitParameters.default_initial_values.copy()
        fixed_values = FitParameters.default_fixed_values.copy()

        default_fraction_value = initial_values.pop('f_px')
        default_fraction_fix = fixed_values.pop('f_px')

        for key in self._parameter_keys:
            if key not in ('dx', 'dy', 'phi', 'total_cts', 'sigma'):
                # assuming a pattern fraction
                initial_values[key] = min(default_fraction_value, 0.5 / self._n_sites)
                fixed_values[key] = default_fraction_fix

        return initial_values, fixed_values

    def _compute_bounds(self):
        """
        Static method to compute the fit bounds
        :return: bounds dictionary
        """

        # Compute defaults
        bounds = FitParameters.default_bounds.copy()
        default_fraction_bounds = bounds.pop('f_px')  # remove 'f_px'
        bounds_temp = dict()
        for i in range(self._n_sites):
            fraction_key = 'f_p' + str(i + 1)  # 'f_p1', 'f_p2', 'f_p3',...
            bounds_temp[fraction_key] = default_fraction_bounds

        bounds.update(bounds_temp)
        return bounds

    def _compute_step_modifier(self):
        """
        Static method to compute the fit step modifiers (scale)
        :return: step modifier dictionary
        """

        # Compute defaults
        scale = FitParameters.default_scale.copy()
        default_fraction_scale = scale.pop('f_px')  # remove 'f_px'
        scale_temp = dict()
        for i in range(self._n_sites):
            fraction_key = 'f_p' + str(i + 1)  # 'f_p1', 'f_p2', 'f_p3',...
            scale_temp[fraction_key] = default_fraction_scale
        scale.update(scale_temp)  # Join dictionaries

        return scale

    def reset_to_defaults(self):
        # Scale and Bounds
        self._scale = self._compute_step_modifier()
        self._bounds = self._compute_bounds()

        # overwrite defaults from Fit
        self._initial_values, self._fixed_values = self._compute_initial_values()

    def update_bounds_with_datapattern(self, datapattern: DataPattern):
        """
        Update the current bounds with the DataPattern mesh information.
        :param datapattern: DataPattern object.
        :return:
        """

        if not isinstance(datapattern, DataPattern):
            raise ValueError('datapattern must be of type DataPattern.')

        dp_bounds_dx = (np.round(datapattern.xmesh[0, 0], 2),
                        np.round(datapattern.xmesh[0, -1], 2))
        dp_bounds_dy = (np.round(datapattern.ymesh[0, 0], 2),
                        np.round(datapattern.ymesh[-1, 0], 2))

        # Get the most limiting bound
        new_bounds_dx = (max(self._bounds['dx'][0], dp_bounds_dx[0]),
                         min(self._bounds['dx'][1], dp_bounds_dx[1]))
        new_bounds_dy = (max(self._bounds['dy'][0], dp_bounds_dy[0]),
                         min(self._bounds['dy'][1], dp_bounds_dy[1]))

        self._bounds['dx'] = new_bounds_dx
        self._bounds['dy'] = new_bounds_dy

    def update_initial_values_with_datapattern(self, datapattern: DataPattern):
        """
        Update the current initial values with the DataPattern information.
        :param datapattern: DataPattern object.
        :return:
        """

        if not isinstance(datapattern, DataPattern):
            raise ValueError('datapattern must be of type DataPattern.')

        self._initial_values['dx'] = datapattern.center[0]
        self._initial_values['dy'] = datapattern.center[1]
        self._initial_values['phi'] = datapattern.angle
        self._initial_values['total_cts'] = datapattern.pattern_matrix.sum()

    def change_initial_values(self, **kwargs):
        """
        Change selected initial values with a user defined value. If the parameter is fixed it will seize to be.
        :param kwargs: possible arguments are 'dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3'
        """
        # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        for key in kwargs.keys():
            if key in self._parameter_keys:
                self._initial_values[key] = kwargs[key]
                self._fixed_values[key] = False
            else:
                raise(ValueError, 'key word ' + key + 'is not recognized!' +
                      '\n Valid keys are, \'dx\',\'dy\',\'phi\',\'total_cts\',\'sigma\',\'f_p1\',\'f_p2\',\'f_p3\'')

    def change_fixed_values(self, **kwargs):
        """
        If the argument value is a numeral, change selected initial values with a user defined fixed value.
        If the argument value is a bool, change if the parameter is fixed or not.
        :param kwargs: Possible arguments are 'dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3'.
                        kw argument values should be bool or a numeral.
        """
        # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        for key in kwargs.keys():
            if key in self._parameter_keys:
                if isinstance(kwargs[key], bool):
                    self._fixed_values[key] = kwargs[key]
                elif isinstance(kwargs[key], (int, float, np.integer, np.float)):
                    self._initial_values[key] = kwargs[key]
                    self._fixed_values[key] = True
                else:
                    raise ValueError(f'{key} argument value needs to be of type bool or float. '
                                     f'{key} is of type {type(kwargs[key])} instead.')

            else:
                raise ValueError('key word ' + key + 'is not recognized! \n' +
                      'Valid keys are, \'dx\',\'dy\',\'phi\',\'total_cts\',\'sigma\',\'f_p1\',\'f_p2\',\'f_p3\'')

    def change_bounds(self, **kwargs):
        """
        Set bounds to a paramater. Bounds are a tuple with two values, for example, (0, None).
        :param kwargs: possible arguments are 'dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3'
        """
        # ('dx', 'dy', 'phi', 'total_cts', 'sigma', 'f_p1', 'f_p2', 'f_p3')
        for key in kwargs.keys():
            if key in self._parameter_keys:
                if not isinstance(kwargs[key], tuple) or len(kwargs[key]) != 2:
                    raise ValueError('Bounds must be a tuple of length 2.')
                self._bounds[key] = kwargs[key]
            else:
                raise ValueError('key word ' + key + 'is not recognized! \n' +
                      'Valid keys are, \'dx\',\'dy\',\'phi\',\'total_cts\',\'sigma\',\'f_p1\',\'f_p2\',\'f_p3\'')

    def change_step_modifier(self, **kwargs):
        """
        Set a step modifier value for a parameter.
        If a modifier of 10 is used for parameter P the fit will try step 10x the default step.
        For the L-BFGS-B minimization method the default steps are 1 for each value exept for the total counts
        that is the order of magnitude of the counts in the data pattern
        :param kwargs: possible arguments are 'dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3'
        """
        # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
        for key in kwargs.keys():
            if key in self._parameter_keys:
                self._scale[key] = kwargs[key]
            else:
                raise ValueError('key word ' + key + 'is not recognized! \n' +
                      'Valid keys are, \'dx\',\'dy\',\'phi\',\'total_cts\',\'sigma\',\'f_p1\',\'f_p2\',\'f_p3\'')

    def get_p0_pfix(self):

        p0 = ()
        pfix = ()

        for key in self._parameter_keys:
            p0 += (self._initial_values[key],)
            pfix += (self._fixed_values[key],)

        return p0, pfix

    def get_scale_for_fit(self, cost_function):

        if cost_function not in ('chi2', 'ml'):
            raise ValueError('cost_function must be \'chi2\' or \'ml\'.')

        scale = ()
        for key in self._parameter_keys:
            # ('dx','dy','phi','total_cts','sigma','f_p1','f_p2','f_p3')
            # total_cts is a spacial case at it uses the counts from the pattern
            if key == 'total_cts':
                if cost_function == 'chi2':
                    patt_sum = self._initial_values['total_cts']  # Expected as the sum of the datapattern.
                    counts_ordofmag = 10 ** (int(math.log10(patt_sum)))
                    scale += (counts_ordofmag * self._scale[key],)
                elif cost_function == 'ml':
                    scale += (-1,)
            else:
                scale += (self._scale[key],)
        return scale
