from distutils.core import setup
import re, io

# https://stackoverflow.com/a/17638236/2707733

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    io.open('main_package/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

setup(
    name='pyfdd',
    version=__version__,
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
