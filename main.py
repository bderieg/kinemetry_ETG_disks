import numpy as np
import matplotlib.pyplot as plt
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
        'velmap_filename' : None,
        'fluxmap_filename' : None,
        'velmap_unc_filename' : None,
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
        'rangeq' : [0.2, 1.0],
        'rangepa' : [-90, 90],
        'nq' : 21,
        'npa' : 21,
        'vsys' : None,
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

dependencies = {
        'flux_cutoff' : [('fluxmap_filename', "\'flux_cutoff\' specified but not \'fluxmap\'; \'flux_cutoff\' will be useless")],
        'saveplots' : [('saveloc', "\'saveplots\' specified but nothing will be saved because \'saveloc\' was not specified")],
        'savedata' : [('saveloc', "\'savedata\' specified but nothing will be saved because \'saveloc\' was not specified")],
        '_MANDATORY' : [(['velmap_filename'], "Could not perform kinemetry because no velocity map was specified")]
    }

##################################
# Attempt to read parameter file #
##################################

# Get parameter file name from command line argument
assert len(sys.argv) == 2, "Incorrect number of command-line arguments (should be 1: parameter file name)"
param_filename = sys.argv[1]

# Read file
params = func.read_properties(param_filename)

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

###############################################
# Use bin map or velocity map from fits file? #
###############################################

usefits = True
if 'binmap_filename' in original_params:
    usefits = False
    if 'velmap_filename' in original_params:
        logging.warning("\'velmap_filename\' and \'binmap_filename\' were both specified . . . defaulting to latter")

#########################################
# Import map data (if using fits files) #
#########################################

# Import images from fits
velmap = fits.open(params['velmap_filename'])[0].data
if params['fluxmap_filename'] is not None:
    fluxmap = fits.open(params['fluxmap_filename'])[0].data
else:  # If fluxmap not specified
    fluxmap = np.asarray(list(map(lambda x:list(map(lambda row:1e9,x)), velmap)))  # list of arbitrary values of same size as velmap
if params['velmap_unc_filename'] is not None:
    velmap_unc = fits.open(params['velmap_unc_filename'])[0].data
else:  # If velmap_unc not specified
    velmap_unc = np.asarray(list(map(lambda x:list(map(lambda row:1.0,x)), velmap)))  # list of arbitrary values of same size as velmap

# None-ify empty pixels
for row in range(len(velmap)):
    for col in range(len(velmap[row])):
        if velmap[row][col] == 0.0:
            velmap[row][col] = None
            fluxmap[row][col] = None
            velmap_unc[row][col] = None

# None-ify unreliable pixels
fluxmap_size = len(fluxmap)*len(fluxmap[0])
num_noneified = 0.0
for row in range(len(fluxmap)):
    for col in range(len(fluxmap[row])):
        if fluxmap[row][col] < params['flux_cutoff']:
            velmap[row][col] = None
            fluxmap[row][col] = None
            velmap_unc[row][col] = None
            num_noneified += 1.0

# Warn if too many pixels are none-ified
if num_noneified/fluxmap_size > 0.98:
    logging.warning('\'flux_cutoff\' is set very high, so not many values are passed to kinemetry; this could result in kinemetry failure')

# Find flux center if necessary
if params['center_method'] != 'fixed':
    params['x0'], params['y0'] = func.centroid(fluxmap)

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
    velmap_unc = np.multiply(velmap, combined_pix_mask)

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
velbin_unc_nan = velmap_unc.ravel()

# Make new lists sans nan values
xbin = []
ybin = []
velbin = []
fluxbin = []
velbin_unc = []
for itr in range(len(velbin_nan)):
    if velbin_nan[itr] == velbin_nan[itr]:
        xbin.append(xbin_nan[itr])
        ybin.append(ybin_nan[itr])
        velbin.append(velbin_nan[itr])
        fluxbin.append(fluxbin_nan[itr])
        velbin_unc.append(velbin_unc_nan[itr])

# Convert to arrays
xbin = np.asarray(xbin)
ybin = np.asarray(ybin)
velbin = np.asarray(velbin)
fluxbin = np.asarray(fluxbin)
velbin_unc = np.asarray(velbin_unc)

#####################################
# Interpolate between bin centroids #
#####################################

# TODO: Update velbin_unc for having interpolated?

# Rectonstruct bin centroids
xcent = []
ycent = []
velcent = []
velunccent = []
fluxcent = []
## Iterate over bins
for f in np.unique(fluxbin):
    binmask = fluxbin==f
    # Find current bin centroids
    cur_xcent = np.mean(xbin[binmask])
    cur_ycent = np.mean(ybin[binmask])
    cur_velcent = np.mean(velbin[binmask])
    cur_velunccent = np.mean(velbin_unc[binmask])
    # Update centroid lists
    xcent.append(cur_xcent)
    ycent.append(cur_ycent)
    velcent.append(cur_velcent)
    velunccent.append(cur_velunccent)
    fluxcent.append(f)

# Update values for interpolation
xbin_pix = xbin.copy()
ybin_pix = ybin.copy()
xbin = np.asarray(xcent)
ybin = np.asarray(ycent)
velbin = np.asarray(velcent)
fluxbin = np.asarray(fluxcent)

#################################################

# Interpolate (RBF)
xyi = np.transpose(np.concatenate([[xbin],[ybin]]))
rbf_flux = interp.RBFInterpolator(xyi, fluxbin, kernel='multiquadric', epsilon=20)
rbf_vel = interp.RBFInterpolator(xyi, velbin, kernel='cubic')
outxy = np.transpose(np.concatenate([[xbin_pix,ybin_pix]]))
flux_interp = rbf_flux(outxy)
vel_interp = rbf_vel(outxy)

# Plot to make sure it looks good
outgrid = np.zeros([len(velmap[0]),len(velmap)])
for n,row,col in zip(range(len(vel_interp)),xbin_pix,ybin_pix):
    outgrid[row,col] = vel_interp[n]

# Export maps to fits
fits.writeto('/home/ben/Desktop/ngc1332_residuals/ngc1332_rbfcubic_mom1.fits', np.flipud(np.rot90(outgrid)), overwrite=True)

# Find residuals and export map
## Import unbinned map
unbinned_map = np.rot90(np.fliplr(fits.open('/home/ben/Desktop/mom1_approx.fits')[0].data))
## Find residuals
residuals = np.abs(outgrid - unbinned_map)
plt.imshow(residuals)
plt.show()
## Export residuals
fits.writeto('/home/ben/Desktop/ngc1332_residuals/ngc1332_rbfcubic_mom1_residuals.fits', np.flipud(np.rot90(residuals)), overwrite=True)

# Update values for kinemetry
xbin = xbin_pix.copy()
ybin = ybin_pix.copy()
fluxbin = flux_interp
velbin = vel_interp

#################################################

# # Interpolate (linear)
# xyi = np.transpose(np.concatenate([[xbin],[ybin]]))
# rbf_flux = interp.LinearNDInterpolator(xyi, fluxbin)
# rbf_vel = interp.LinearNDInterpolator(xyi, velbin)
# outxy = np.transpose(np.concatenate([[xbin_pix,ybin_pix]]))
# flux_interp = rbf_flux(outxy)
# vel_interp = rbf_vel(outxy)
# 
# # Plot to make sure it looks good
# outgrid = np.zeros([len(velmap[0]),len(velmap)])
# for n,row,col in zip(range(len(vel_interp)),xbin_pix,ybin_pix):
#     outgrid[row,col] = vel_interp[n]
# 
# # Export maps to fits
# fits.writeto('/home/ben/Desktop/ngc1332_residuals/ngc1332_linear_mom1.fits', np.flipud(np.rot90(outgrid)), overwrite=True)
# 
# # Find residuals and export map
# ## Import unbinned map
# unbinned_map = np.rot90(np.fliplr(fits.open('/home/ben/Desktop/mom1_approx.fits')[0].data))
# ## Find residuals
# residuals = np.abs(outgrid - unbinned_map)
# plt.imshow(residuals)
# plt.show()
# ## Export residuals
# fits.writeto('/home/ben/Desktop/ngc1332_residuals/ngc1332_linear_mom1_residuals.fits', np.flipud(np.rot90(residuals)), overwrite=True)
# 
# # Update values for kinemetry
# xbin = xbin_pix.copy()
# ybin = ybin_pix.copy()
# fluxbin = flux_interp
# velbin = vel_interp

#################################################

# # Interpolate (griddata)
# xyi = np.transpose(np.concatenate([[xbin],[ybin]]))
# outxy = np.transpose(np.concatenate([[xbin_pix,ybin_pix]]))
# flux_interp = interp.griddata(xyi, fluxbin, outxy)
# vel_interp = interp.griddata(xyi, velbin, outxy, method='linear')
# 
# # Plot to make sure it looks good
# outgrid = np.zeros([len(velmap[0]),len(velmap)])
# for n,row,col in zip(range(len(vel_interp)),xbin_pix,ybin_pix):
#     outgrid[row,col] = vel_interp[n]
# 
# # Export maps to fits
# fits.writeto('/home/ben/Desktop/ngc1332_residuals/ngc1332_gridlinear_mom1.fits', np.flipud(np.rot90(outgrid)), overwrite=True)
# 
# # Find residuals and export map
# ## Import unbinned map
# unbinned_map = np.rot90(np.fliplr(fits.open('/home/ben/Desktop/mom1_approx.fits')[0].data))
# ## Find residuals
# residuals = np.abs(outgrid - unbinned_map)
# plt.imshow(residuals)
# plt.show()
# ## Export residuals
# fits.writeto('/home/ben/Desktop/ngc1332_residuals/ngc1332_gridlinear_mom1_residuals.fits', np.flipud(np.rot90(residuals)), overwrite=True)
# 
# # Update values for kinemetry
# xbin = xbin_pix.copy()
# ybin = ybin_pix.copy()
# fluxbin = flux_interp
# velbin = vel_interp

#################################################

# # Krige
# krig_flux = uk(xbin, ybin, fluxbin, variogram_model='hole-effect', enable_plotting=False, exact_values=True, pseudo_inv=True)
# krig_vel = uk(xbin, ybin, velbin, variogram_model='exponential', exact_values=True)
# krig_flux_points, sig = krig_flux.execute('points', xbin_pix.astype(float), ybin_pix.astype(float))
# krig_vel_points, sig = krig_vel.execute('points', xbin_pix.astype(float), ybin_pix.astype(float))
# 
# # Plot to make sure it looks good
# outgrid = np.zeros([len(velmap[0]),len(velmap)])
# for n,row,col in zip(range(len(krig_vel_points)),xbin_pix,ybin_pix):
#     outgrid[row,col] = krig_vel_points[n]
# 
# # Export maps to fits
# fits.writeto('/home/ben/Desktop/ngc1332_residuals/ngc1332_krigeexp_mom1.fits', np.flipud(np.rot90(outgrid)), overwrite=True)
# 
# # Find residuals and export map
# ## Import unbinned map
# unbinned_map = np.rot90(np.fliplr(fits.open('/home/ben/Desktop/mom1_approx.fits')[0].data))
# ## Find residuals
# residuals = np.abs(outgrid - unbinned_map)
# plt.imshow(residuals)
# plt.show()
# ## Export residuals
# fits.writeto('/home/ben/Desktop/ngc1332_residuals/ngc1332_krigeexp_mom1_residuals.fits', np.flipud(np.rot90(residuals)), overwrite=True)
# 
# # Update values for kinemetry
# xbin = xbin_pix.copy()
# ybin = ybin_pix.copy()
# fluxbin = krig_flux_points
# velbin = krig_vel_points

##############################
# Do the main kinemetry task #
##############################

k = kin.kinemetry(xbin=xbin, ybin=ybin, moment=velbin, error=velbin_unc,
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
