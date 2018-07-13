from distutils.core import setup

setup(
    name='pyfdd',
    version='0.0.01',
    packages=['pyfdd', 'pyfdd.lib2dl', 'pyfdd.datapattern', 'pyfdd.datapattern.CustomWidgets', 'scripts', 'my_scripts',
              'test_pyfdd', 'ecsli_tools'],
    url='https://github.com/eric-presbitero/pyfdd',
    license='GPL-3.0',
    author='E David-Bosne',
    author_email='eric.bosne@cern.ch',
    description='Software for fitting channelling data for lattice location.'
)
