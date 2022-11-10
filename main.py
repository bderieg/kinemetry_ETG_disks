import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from plotbin.plot_velfield import plot_velfield
import pandas as pd
import time
from astropy.io import fits
import csv
import operator
from regions import Regions
import regions

from kinemetry_scripts.kinemetry import kinemetry
import kinemetry_scripts.kinemetry as kin

import plotting_scripts as plotter

import sys
import warnings
import logging
# warnings.filterwarnings(action='ignore', category=UserWarning)  # Comment to allow warnings

###########################
# Define global constants #
###########################
default_params = {  # If None, then no default exists--user must define in parameter file
        'velmap_filename' : None,
        'fluxmap_filename' : None,
        'ntrm' : 6,
        'scale' : 1,
        'center_method' : 'free',  # 'free', 'fixed', 'fc'
        'x0' : 0,
        'y0' : 0,
        'nrad' : 100,
        'allterms' : False,
        'even' : False,
        'cover' : 1,
        'plot' : True,
        'rangeq' : [0.2, 1.0],
        'rangepa' : [-90, 90],
        'vsys' : 0,
        'drad' : 1,
        'incrad' : 1,
        'flux_cutoff' : 0.0,
        'saveloc' : 'none',
        'objname' : 'noname',
        'verbose' : False,
        'ring' : 0.0,
        'saveplots' : False,
        'savedata' : False,
        'badpixel_filename' : 'none'
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
                raise csv.Error("Parameter file syntax error on line "+str(row))
            try:  # Convert data types except for strings
                row[1] = eval(row[1])
            except SyntaxError:
                pass
            result[row[0].lower()] = row[1]  # Assign row to dictionary

    return result

##############################################
# Define function to find centroid of a list #
##############################################
def centroid(img):
    weighted_flux_x = 0.0
    weighted_flux_y = 0.0
    total_flux = 0.0
    
    for row in range(len(img)):
        for col in range(len(img[row])):
            if img[row][col] == img[row][col]:
                weighted_flux_x += col*img[row][col]
                weighted_flux_y += row*img[row][col]
                total_flux += img[row][col]

    xc = weighted_flux_x/total_flux
    yc = weighted_flux_y/total_flux

    return xc, yc

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
if params['vsys'] == 0:
    params['vsys'] = None
assert params['center_method'] in {'free', 'fixed', 'fc'}, "\'center_method\' argument value invalid"
if params['center_method'] == 'free':
    params['fixcen'] = False
else:
    params['fixcen'] = True

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
for row in range(len(fluxmap)):
    for col in range(len(fluxmap[row])):
        if fluxmap[row][col] < params['flux_cutoff']:
            velmap[row][col] = None
            fluxmap[row][col] = None

# Find flux center if necessary
if params['center_method'] != 'fixed':
    params['x0'], params['y0'] = centroid(fluxmap)

# Mask bad pixels from input file
if params['badpixel_filename'] != 'none':
    # Read region file
    regions = Regions.read(params['badpixel_filename'], format='ds9')
    # Make pixel masks
    pix_mask_list = []
    for reg in regions:
        pix_mask_list.append(reg.to_mask(mode='center'))
    # Combine all masks
    combined_pix_mask = np.zeros(velmap.shape)
    for mask in pix_mask_list:
        combined_pix_mask += mask.to_image(velmap.shape)
    # Flip mask values
    combined_pix_mask = 1 - combined_pix_mask
    # Make 0s None
    for row in range(len(combined_pix_mask)):
        for col in range(len(combined_pix_mask[row])):
            if combined_pix_mask[row][col] == 0:
                combined_pix_mask[row][col] = None
    # Multiply with velmap
    velmap = np.multiply(velmap, combined_pix_mask)

# Make mask image
value_mask = fluxmap.copy()
for row in range(len(fluxmap)):
    for col in range(len(fluxmap[row])):
        if fluxmap[row][col] > 0 and velmap[row][col] > 0:
            value_mask[row][col] = 1e-5
        else:
            value_mask[row][col] = 1

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
        ntrm=params['ntrm'], incrad=params['incrad'],
        fixcen=params['fixcen'], nrad=params['nrad'],
        allterms=params['allterms'], even=params['even'],
        cover=params['cover'], plot=params['plot'],
        vsys=params['vsys'], drad=params['drad'],
        ring=params['ring']/params['scale'], verbose=params['verbose']
        )

# Plot radial profiles
radial_data = plotter.plot_kinemetry_profiles(k, params['scale'])
if (params['saveloc'] != 'none') and params['saveplots']:
    plt.savefig(params['saveloc']+params['objname']+'_radial_profiles.png', dpi=1000)

# Plot v_los maps
spatial_data = plotter.plot_vlos_maps(xbin, ybin, velbin, k, value_mask=value_mask)
if (params['saveloc'] != 'none') and params['saveplots']:
    plt.savefig(params['saveloc']+params['objname']+'_velocity_maps.png', dpi=1000)

# If not saving figures, just show
if not params['saveloc']:
    plt.show()

if params['saveloc'] and not (params['saveplots'] or params['savedata']):
    logging.warning('Save location set but nothing to save! Did you mean to set \'saveplots\' or \'savedata\'?')

# Save data
if params['savedata']:
    radial_data.to_csv(params['saveloc']+params['objname']+'_radial_data.csv', index=True)
    spatial_data.to_csv(params['saveloc']+params['objname']+'_spatial_data.csv', index=False)
