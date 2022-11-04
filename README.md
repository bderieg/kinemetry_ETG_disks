# kinemetry_ETG_disks
## Description
A set of scripts using Davor Krajnovic's "kinemetry" program for Python. Specifically for doing moment 1 kinemetry on thin disks from ALMA spectral data.

## Parameter Files
The user should create his/her own parameter files for each target. Parameter files should be of the format key=value, one per line, with the following mandatory keys
- velmap_filename
- fluxmap_filename

and the following optional kinemetry parameters (see documentation there)
- ntrm
- scale
- x0
- y0
- fixcen
- nrad
- allterms
- even
- cover
- plot
- rangeq
- rangepa
- ring
- verbose

and the following other optional parameters
- flux_cutoff
    - Velocity values that spatially correspond to flux values below this level will be thrown out before running kinemetry
- obj_name
    - A string with the object name; used for file naming if 'save_loc' is set
- save_loc
    - A string containing a file save location
- badpixel_filename
    - File name of a DS9 .reg file containing regions to mask when performing kinemetry

See the existing parameter files as examples.

## Running Kinemetry
With parameter files specified, kinemetry can be run with
```
python3 main.py [parameter file]
```
(or equivalent command with whatever alias python is set to). Optionally, the user can run the 'run_all.sh' script, which will automatically run kinemetry on all the parameter files ending in ".param" which are located in the "param_files" folder.
