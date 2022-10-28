import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from plotbin.plot_velfield import plot_velfield
import pandas as pd
import time
from astropy.io import fits
import csv
import operator

from kinemetry_scripts.kinemetry import kinemetry
import kinemetry_scripts.kinemetry as kin

import plotting_scripts as plotter

import sys

###########################
# Define global constants #
###########################
default_params = {  # If None, then no default exists--user must define in parameter file
        'velmap_filename' : None,
        'fluxmap_filename' : None,
        'ntrm' : 6,
        'scale' : 1,
        'x0' : 0,
        'y0' : 0,
        'fixcen' : True,
        'nrad' : 100,
        'allterms' : False,
        'even' : False,
        'cover' : 1,
        'plot' : True,
        'rangeq' : [0.2, 1.0],
        'rangepa' : [-90, 90],
        'saveloc' : 'none',
        'objname' : 'noname'
    }

#######################################
# Define parameter file read function #
#######################################

def read_properties(filename):
    """ Reads a given properties file with each line of the format key=value.  Returns a dictionary containing the pairs.

    Keyword arguments:
        filename -- the name of the file to be read
    """
    result = {}
    with open(filename, "r") as csvfile:  # Open file as read-only
        # Define file-read object
        reader = csv.reader(csvfile, delimiter='=', escapechar='\\', quoting=csv.QUOTE_NONE)

        # Iterate through rows in file
        for row in reader:
            row = [i.replace(" ","") for i in row]
            if len(row) == 0:  # If blank row
                continue
            elif len(row) != 2:  # If row doesn't make sense
                raise csv.Error("Too many fields on row with contents: "+str(row))
            try:  # Convert data types except for strings
                row[1] = eval(row[1])
            except SyntaxError:
                pass
            result[row[0].lower()] = row[1]  # Assign row to dictionary

    return result

##################################
# Attempt to read parameter file #
##################################

# Get parameter file name from command line argument
assert len(sys.argv) == 2, "Incorrect number of arguments (should be 1--parameter file name)"
param_filename = sys.argv[1]

# Read file
params = read_properties(param_filename)

###################################################
# Fill params with default values if not explicit #
###################################################

for key in default_params:
    if key not in params:
        assert default_params[key] is not None, "Mandatory argument not specified in parameter file: " + key
        params[key] = default_params[key]

###################
# Import map data #
###################

# Import images from fits
velmap = fits.open(params['velmap_filename'])[0].data
fluxmap = fits.open(params['fluxmap_filename'])[0].data

# None-ify empty pixels
for row in range(len(velmap)):
    for col in range(len(velmap[row])):
        if velmap[row][col] == 0.0:
            velmap[row][col] = None
            fluxmap[row][col] = None

# None-ify unreliable pixels
maxflux = np.asarray(fluxmap).max()
for row in range(len(fluxmap)):
    for col in range(len(fluxmap[row])):
        if fluxmap[row][col] < 0.29*maxflux:
            velmap[row][col] = None
            fluxmap[row][col] = None

# Convert to 1D arrays
ny,nx = velmap.shape
x = (np.arange(0,nx))
y = (np.arange(0,ny))
xx, yy = np.meshgrid(x, y)
xbin_nan = xx.ravel()
ybin_nan = yy.ravel()
velbin_nan = velmap.ravel()
fluxbin_nan = fluxmap.ravel()

# Make new lists sans nan values
xbin = []
ybin = []
velbin = []
fluxbin = []
for itr in range(len(velbin_nan)):
    if velbin_nan[itr] == velbin_nan[itr]:
        xbin.append(xbin_nan[itr])
        ybin.append(ybin_nan[itr])
        velbin.append(velbin_nan[itr])
        fluxbin.append(fluxbin_nan[itr])

# Convert to arrays
xbin = np.asarray(xbin)
ybin = np.asarray(ybin)
velbin = np.asarray(velbin)
fluxbin = np.asarray(fluxbin)

##############################
# Do the main kinemetry task #
##############################

k = kin.kinemetry(xbin=xbin, ybin=ybin, moment=velbin,
        x0=params['x0'], y0=params['y0'],
        rangeQ=params['rangeq'], rangePA=params['rangepa'],
        ntrm=params['ntrm'], scale=params['scale'],
        fixcen=params['fixcen'], nrad=params['nrad'],
        allterms=params['allterms'], even=params['even'],
        cover=params['cover'], plot=params['plot']
        )

plotter.plot_kinemetry_profiles(k)
if params['saveloc'] != 'none':
    plt.savefig(params['saveloc']+params['objname']+'_radial_profiles.png', dpi=1000)
plotter.plot_vlos_maps(xbin, ybin, velbin, k)
if params['saveloc'] != 'none':
    plt.savefig(params['saveloc']+params['objname']+'_velocity_maps.png', dpi=1000)
plotter.plot_flux_vel(fluxbin, velbin)
if params['saveloc'] != 'none':
    plt.savefig(params['saveloc']+params['objname']+'_flux_velocity_hist.png', dpi=1000)
else:
    plt.show()
