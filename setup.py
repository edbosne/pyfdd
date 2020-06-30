import io
import re
from distutils.core import setup

# https://stackoverflow.com/a/17638236/2707733

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    io.open('pyfdd/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

setup(
    name='pyfdd',
    version=__version__,
    packages=['pyfdd', 'pyfdd.core', 'pyfdd.core.lib2dl', 'pyfdd.core.datapattern', 'examples',
              'ecsli_tools'],
    install_requires=[
          'numpy', 'matplotlib', 'scipy', 'numdifftools', 'pandas', 'seaborn', 'PyQt5==5.9.2'],
    python_requires='>=3',
    url='https://github.com/eric-presbitero/pyfdd',
    license='GPL-3.0',
    author='E David-Bosne',
    author_email='eric.bosne@cern.ch',
    description='Software for fitting channelling data for lattice location.'
)
