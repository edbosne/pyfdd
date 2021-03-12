import configparser
import os
import json
import warnings
from PyQt5 import QtCore


parser = None
filename = 'config.ini'
path = './'


def parser_not_set(*args):
    raise ValueError('Parser not set.')


# expose built-in methods
get = parser_not_set
getboolean = parser_not_set
getfloat = parser_not_set
getint = parser_not_set
getlist = parser_not_set
getdict = parser_not_set


class QtSignals(QtCore.QObject):
    # make an updated signal to propagate changes to other datapattern widgets
    updated = QtCore.pyqtSignal()


signals = QtSignals()


def read():
    """
    Loads configuration file or creates a new is necessary
    :return:
    """
    global parser
    global filename
    global path

    # expose built-in methods
    global get
    global getboolean
    global getfloat
    global getint
    global getlist
    global getdict

    # Create parser
    if isinstance(parser, configparser.ConfigParser):
        warnings.warn('Parser is already created. Overwriting.')
    parser = configparser.ConfigParser()
    config_fullname = os.path.join(path, filename)
    parser.read(config_fullname)

    # expose built-in methods - redefinition
    get = parser.get
    getboolean = parser.getboolean
    getfloat = parser.getfloat
    getint = parser.getint

    # Use JSON to parse lists and dicts that are saved as strings
    def new_getlist(section, option):
        return json.loads(get(section, option))
    getlist = new_getlist

    def new_getdict(section, option):
        return json.loads(get(section, option))
    getdict = new_getdict


def write():
    """
    saves configuration to file
    :return:
    """
    global parser
    global filename
    global path

    global signals

    if parser is None:
        raise ValueError('Trying to write configuration file but the parser is not set')
    else:
        assert isinstance(parser, configparser.ConfigParser)

    config_fullname = os.path.join(path, filename)
    with open(config_fullname, 'w') as configfile:
        parser.write(configfile)

    signals.updated.emit()


def load_config_option(self, section, optname, opttype=None):
    """
    Loads a config option from a defined section. If it is not defined it uses the hardcoded value.
    :param self:
    :param section:
    :param optname:
    :param opttype:
    :return:
    """
    if parser is None:
        raise ValueError('Trying to write configuration file but the parser is not set')
    else:
        # print({section: dict(parser[section]) for section in parser.sections()})
        if parser.has_option(section, optname):
            if get(section, optname) == 'None':
                vars(self)[optname] = None
            elif opttype is None:
                vars(self)[optname] = get(section, optname)
            elif opttype is 'bool':
                vars(self)[optname] = getboolean(section, optname)
            elif opttype is 'int':
                vars(self)[optname] = getint(section, optname)
            elif opttype is 'float':
                vars(self)[optname] = getfloat(section, optname)
            elif opttype is 'list':
                vars(self)[optname] = getlist(section, optname)
            elif opttype is 'dict':
                vars(self)[optname] = getdict(section, optname)
        else:
            if opttype is 'dict':
                # Python dictionaries need to be converted to JSON which uses double quotes
                parser[section][optname] = json.dumps(dict(vars(self)[optname]))
            else:
                parser[section][optname] = str(vars(self)[optname])

