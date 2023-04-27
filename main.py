import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from plotbin.plot_velfield import plot_velfield
import pandas as pd
from astropy.io import fits
import regions
from regions import Regions
import scipy.interpolate as interp
from pykrige.ok import OrdinaryKriging as uk

import kinemetry_scripts.kinemetry as kin
import plotting_scripts as plotter
import other_functions as func

import sys
import warnings
import logging
# warnings.filterwarnings(action='ignore', category=UserWarning)  # Uncomment to supress all warnings

###########################
# Define global constants #
###########################
default_params = {
        'data_filename' : None,
        'ntrm' : 6,
        'scale' : 1,
        'center_method' : 'free',
        'x0' : 0,
        'y0' : 0,
        'nrad' : 100,
        'allterms' : False,
        'even' : False,
        'cover' : 1,
        'plot' : True,
        'plotlimspa' : None,
        'plotlimsq' : None,
        'plotlimsk1' : None,
        'plotlimsk5k1' : None,
        'rangeq' : [0.2, 1.0],
        'rangepa' : [-90, 90],
        'nq' : 21,
        'npa' : 21,
        'vsys' : None,
        'drad' : 1,
        'extrap_pixels' : 0.0,
        'incrad' : 1,
        'flux_cutoff' : 0.0,
        'bad_bins' : [],
        'saveloc' : 'none',
        'objname' : 'noname',
        'verbose' : False,
        'ring' : 0.0,
        'saveplots' : False,
        'savedata' : False
    }

dependencies = {
        'saveplots' : [('saveloc', "\'saveplots\' specified but nothing will be saved because \'saveloc\' was not specified")],
        'savedata' : [('saveloc', "\'savedata\' specified but nothing will be saved because \'saveloc\' was not specified")],
        '_MANDATORY' : [(['data_filename'], "Could not perform kinemetry because no data file was specified")]
    }

##################################
# Attempt to read parameter file #
##################################

# Get parameter file name from command line argument
assert len(sys.argv) == 2, "Incorrect number of command-line arguments (should be 1: parameter file name)"
param_filename = sys.argv[1]

# Read file
params = func.read_properties(param_filename)

# Set variable to parameter file path (for relative paths in the parameter file)
param_filepath = ''.join((param_filename.rpartition("/"))[:-1])
if params['data_filename'][0] != "/":  # If it's not an absolute path
    params['data_filename'] = param_filepath + params['data_filename']
    params['saveloc'] = param_filepath + params['saveloc']

###################################################
# Fill params with default values if not explicit #
###################################################

original_params = params.copy()
for key in default_params:
    if key not in params:
        params[key] = default_params[key]
    if key in dependencies:
        for item in dependencies[key]:
            if item[0] not in original_params:
                logging.warning(item[1])
assert params['center_method'] in {'free', 'fixed', 'fc'}, "\'center_method\' argument value invalid"
if params['center_method'] == 'free':
    params['fixcen'] = False
else:
    params['fixcen'] = True

##########################################
# Give warnings for ill-specified params #
##########################################

if params['rangeq'][0] == 0.0:
    logging.warning('setting 0 as a lower bound for \'rangeq\' may result in kinemetry failure; use a small non-zero value instead')

#####################################################
# Exit cleanly if mandatory arguments not specified #
#####################################################

for item in dependencies['_MANDATORY']:
    if not any([itr in original_params for itr in item[0]]):
        logging.warning(item[1])
        exit()

######################
# Import moment data #
######################

# Import dataframe
moment_data = pd.read_csv(params['data_filename'], skiprows=9)

# Find flux center if necessary
if params['center_method'] != 'fixed':
    params['x0'], params['y0'] = func.centroid(moment_data)

# Remove bad bins
## From flux threshold
below_cutoff = moment_data['mom0 (Jy/beam)'] < params['flux_cutoff']
moment_data = moment_data[~below_cutoff]
## From bad_bins keyword
moment_data.drop([x-10 for x in list(params['bad_bins'])], inplace=True)

xbin = moment_data['x (pix)'].values
ybin = moment_data['y (pix)'].values
velbin = moment_data['mom1 (km/s)'].values
fluxbin = moment_data['mom0 (Jy/beam)'].values
velbin_unc = moment_data['mom1unc (km/s)'].values

#####################################
# Interpolate between bin centroids #
#####################################

# Make output grid
outx, outy = np.meshgrid(np.arange(int(np.min(xbin)),int(np.max(xbin))), np.arange(int(np.min(ybin)),int(np.max(ybin))))
outxy = np.column_stack([outx.ravel(), outy.ravel()])
interp_reg = Path(np.transpose([xbin, ybin]))
inside_mask = interp_reg.contains_points(outxy, radius=params['extrap_pixels'])
outxy = outxy[inside_mask]

# Interpolate (RBF)
xyi = np.transpose(np.concatenate([[xbin],[ybin]]))
rbf_flux = interp.RBFInterpolator(xyi, fluxbin, kernel='thin_plate_spline')
rbf_vel = interp.RBFInterpolator(xyi, velbin, kernel='quintic')
flux_interp = rbf_flux(outxy)
vel_interp = rbf_vel(outxy)

# Update values for kinemetry
xbin = outxy[:,0]
ybin = outxy[:,1]
fluxbin = flux_interp
velbin = vel_interp

##############################
# Do the main kinemetry task #
##############################

k = kin.kinemetry(xbin=xbin, ybin=ybin, moment=velbin, #error=velbin_unc,
        x0=params['x0'], y0=params['y0'],
        rangeQ=params['rangeq'], rangePA=params['rangepa'],
        nq=params['nq'], npa=params['npa'],
        ntrm=params['ntrm'], incrad=params['incrad'],
        fixcen=params['fixcen'], nrad=params['nrad'],
        allterms=params['allterms'], even=params['even'],
        cover=params['cover'], plot=params['plot'],
        vsys=params['vsys'], drad=params['drad'],
        ring=params['ring']/params['scale'], verbose=params['verbose']
        )

# Plot radial profiles
radial_data = plotter.plot_kinemetry_profiles(k, params['scale'], 
        user_plot_lims={
            "pa":params["plotlimspa"],
            "q":params["plotlimsq"],
            "k1":params["plotlimsk1"],
            "k5k1":params["plotlimsk5k1"]}
        )
if (params['saveloc'] != 'none') and params['saveplots']:
    plt.savefig(params['saveloc']+params['objname']+'_radial_profiles.png', dpi=1000)

# Plot v_los maps
spatial_data = plotter.plot_vlos_maps(xbin, ybin, velbin, k)
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
