# MRR_Processing
A pyhon module to run the data processing of micro-ring resonator spectra.

## Overview

'__utils.py__' contains utility functions,
'__scripts.py__' contains scripts for larger tasks like plotting and fitting data, and '__Full_Ring_Processing.py__' is where the processing is done.

## How to use

Inside '__Full_Ring_Processing.py__' The user should point to the raw data file and specify their own parameters inside the CONFIG dictionary. 

The code will find the resonances in the spectrum, fit them each of them with a Lorentzian, and extract the relevant resonace data. 

Then, optionally further calculations can be done such as FSR, group index, etc..

All the data and figures are saved to the '__Figures__' folder.

If you have questions or suggestions please email me at jonkere.student.ubc.ca.
