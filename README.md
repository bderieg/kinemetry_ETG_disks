# kinemetry_ETG_disks
## Description
A set of scripts using Davor Krajnovic's "kinemetry" program for Python. Specifically for doing moment 1 kinemetry on thin disks from ALMA spectral data.

## Parameter Files
The user should create his/her own parameter files for each target (see examples folder). Parameter files should be of the format key=value (if value is a string, quotations should be placed around the it), one per line, with the following mandatory keys
- data_filename (csv containing moment/uncertainty data)

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
- extrap_pixels
    - When interpolating, how far outside the data (in pixels) should be extrapolated?
- objname
    - A string with the object name; used for file naming if 'save_loc' is set
- plotlimspa
    - A 2-element list (square brackets) with lower/upper y-limits for plotting the PA
- plotlimsq
    - Same as 'plotlimspa' but for q plot
- plotlimsk1
    - Same as 'plotlimspa' but for k1 plot
- plotlimsk5k1
    - Same as 'plotlimspa' but for k5/k1 plot
- saveloc
    - A string containing a file save location (does not save anything by default; 'saveplots' and/or 'savedata' should also be set)
- saveplots
    - If true, some kinemetry plots will be saved to the location specified by 'saveloc'
- savedata
    - If true, some useful data will be saved in .csv format to the location specified by 'saveloc'
- bad_bins
    - List of (comma-separated) row numbers from data file corresponding to bad data

See the existing parameter files as examples.

### A note on file names
For any of the keywords which specifies a filename, either an absolute path (starting with a "/") or a relative path from the location of the parameter file can be given.

## Running Kinemetry
With parameter files specified, kinemetry can be run with
```
python3 main.py [parameter file]
```
(or equivalent command with whatever python is set to). If the user just wants to run the examples, a shell script titled 'run_all.sh' has been placed in the 'examples' folder for convenience (this runs kinemetry on all the data/parameter files in the 'examples' folder). Note that the 'examples/data' folder is intentionally left empty, to be filled when the user runs these examples.
