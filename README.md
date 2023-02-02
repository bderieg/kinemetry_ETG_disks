# kinemetry_ETG_disks
## Description
A set of scripts using Davor Krajnovic's "kinemetry" program for Python. Specifically for doing moment 1 kinemetry on thin disks from ALMA spectral data.

## Parameter Files
The user should create his/her own parameter files for each target (see examples folder). Parameter files should be of the format key=value (if value is a string, quotations should be placed around the it), one per line, with the following mandatory keys
- velmap_filename
- fluxmap_filename

and the following optional kinemetry parameters (see documentation there)
- ntrm
- scale
- x0
- y0
- nrad
- allterms
- even
- cover
- plot
- rangeq
- rangepa
- nq
- npa
- ring
- vsys
- verbose

and the following other optional parameters
- center_method
    - A string with the following options:
        - 'free': the center will be found separately for each ellipse from the kinemetry best fit
        - 'fixed': the center will be fixed for all ellipses with the user-defined 'x0' and 'y0' arguments
        - 'fc': for 'flux centroid'; the center will be fixed for all ellipses at the flux centroid
- drad
    - Kinemetry will be sampled with this (pixel) spacing in radius. Default is 1.0.
- incrad
    - The spacing between successive kinemetry radii is multiplied by this much. Default is 1.0 (evenly spaced).
- flux_cutoff
    - Velocity values that spatially correspond to flux values below this level will be thrown out before running kinemetry
- objname
    - A string with the object name; used for file naming if 'save_loc' is set
- saveloc
    - A string containing a file save location (does not save anything by default; 'saveplots' and/or 'savedata' should also be set)
- saveplots
    - If true, some kinemetry plots will be saved to the location specified by 'saveloc'
- savedata
    - If true, some useful data will be saved in .csv format to the location specified by 'saveloc'
- badpixel_filename
    - File name of a DS9 .reg file containing regions to mask when performing kinemetry
- velmap_unc_filename
    - File name of moment 1 uncertainty map

See the existing parameter files as examples.

## Running Kinemetry
With parameter files specified, kinemetry can be run with
```
python3 main.py [parameter file]
```
(or equivalent command with whatever alias python is set to). If the user just wants to run the examples, a shell script titled 'run_all.sh' has been placed in the 'examples' folder for convenience (this runs kinemetry on all the data/parameter files in the 'examples' folder). Note that the 'examples/data' folder is intentionally left empty, to be filled when the user runs these examples.
