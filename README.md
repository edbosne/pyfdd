# PyFDD


## General usage notes
===========================

- PyFDD is a software for fitting channeling data for lattice location

- Only library files of type lib2dl are supported

- Use medipix matrix to prepare the data

- Use fit or FitManager for fitting


## Structure

- The DataPattern class is the data class. It holds the measured patter, angular calibration and the necessary tools for edition.

- The lib2dl class reads 2dl libraries and puts them in the format that is necessary for the fit.

- The Fit class does the Fit. If one need to do one single fit for testing this class can be used but for regular analysis the FitManager class is advised.

- The FitManager is the general purpose fit class and the most used in practice. It can fit a pattern over several lattice sites and outputs the results in a .csv file that can be opened with excel.

In the scripts folder one can see examples of how to use the MedipxMatrix to prepare patterns for analysis and then run the analysis with FitManager.


## Contact
For bugs and suggestions please contact:
email.: eric.bosne@cern.ch


## Licence
PyFDD is open source under a GPL3 licence. Please look at the LICENCE.txt file for more info.
