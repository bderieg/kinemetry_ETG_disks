# kinemetry_ETG_disks
## Description
A set of scripts using Davor Krajnovic's "kinemetry" program for Python. Specifically for doing moment 1 kinemetry on thin disks from ALMA spectral data.

## Parameter Files
The user should create his/her own parameter files for each target. Parameter files should be of the format key=value (if value is a string, quotations should be placed around the it), one per line, with the following mandatory keys
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

See the existing parameter files as examples.

## Running Kinemetry
With parameter files specified, kinemetry can be run with
```
python3 main.py [parameter file]
```
(or equivalent command with whatever alias python is set to). Optionally, the user can run the 'run_all.sh' script, which will automatically run kinemetry on all the parameter files ending in ".param" which are located in the "param_files" folder.
