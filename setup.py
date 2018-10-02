from distutils.core import setup

setup(
    name='pyfdd',
    version='0.4',
    packages=['pyfdd', 'pyfdd.lib2dl', 'pyfdd.datapattern', 'pyfdd.datapattern.CustomWidgets', 'examples',
             'ecsli_tools'],
    install_requires=[
          'numpy', 'matplotlib', 'scipy', 'numdifftools', 'pandas'],
    url='https://github.com/eric-presbitero/pyfdd',
    license='GPL-3.0',
    author='E David-Bosne',
    author_email='eric.bosne@cern.ch',
    description='Software for fitting channelling data for lattice location.'
)
