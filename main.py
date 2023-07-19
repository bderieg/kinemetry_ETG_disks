import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from plotbin.plot_velfield import plot_velfield
import pandas as pd
import scipy.interpolate as interp
from scipy.spatial import ConvexHull
import astropy.units as u
import json
from tqdm.auto import tqdm
import corner

import kinemetry_scripts.kinemetry as kin
import plotting_scripts as plotter
import other_functions as func
import read_properties as rp

import sys
import warnings
import logging
# warnings.filterwarnings(action='ignore', category=UserWarning)  # Uncomment to supress all warnings

###########################
# Define global constants #
###########################
default_params = {
        'dataloc' : None,
        'ntrm' : 6,
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
        'nq' : 40,
        'npa' : 40,
        'vsys' : None,
        'drad' : 1,
        'extrap_pixels' : 0.0,
        'incrad' : 1,
        'flux_cutoff' : 0.0,
        'snr_cutoff' : 0.0,
        'bad_bins' : [],
        'saveloc' : 'none',
        'objname' : 'noname',
        'linename' : 'noline',
        'verbose' : False,
        'ring' : 0.0,
        'saveplots' : False,
        'savedata' : False,
        'calc_mass' : False,
        'mc' : False
    }

dependencies = {
        'saveplots' : [('saveloc', "\'saveplots\' specified but nothing will be saved because \'saveloc\' was not specified")],
        'savedata' : [('saveloc', "\'savedata\' specified but nothing will be saved because \'saveloc\' was not specified")],
        '_MANDATORY' : [(['dataloc'], "Could not perform kinemetry because no data location was specified")]
    }

transitions = {
        'a' : 'CO(1-0)',
        'b' : 'CO(2-1)',
        'c' : 'CO(3-2)',
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
if params['dataloc'][0] != "/":  # If it's not an absolute path
    params['dataloc'] = param_filepath + params['dataloc']
    params['saveloc'] = param_filepath + params['saveloc']
if params['dataloc'][-1] != "/":
    params['dataloc'] += "/"
params['data_filename'] = params['dataloc'] + 'mom_bin_vals.csv'

###################################################
# Fill params with default values if not explicit #
###################################################

original_params = params.copy()
for key in default_params:
    if key not in params:
        params[key] = default_params[key]
    if key in dependencies:
        for item in dependencies[key]:
            if item[0] not in original_params and key in original_params:
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

#################################
# Add in spreadsheet parameters #
#################################

params['distance'] = rp.get_prop(params['objname'],'D_L (Mpc)')
params['distance_unc'] = rp.get_prop(params['objname'],'D_L Unc.')
params['redshift'] = rp.get_prop(params['objname'],'Redshift (via NED)')
# pa_data = json.load(open('/home/ben/Desktop/research/research_boizelle_working/ap_phot_data/pa_data.json'))
# q_data = json.load(open('/home/ben/Desktop/research/research_boizelle_working/ap_phot_data/q_data.json'))
# if params['objname'] in pa_data:
#     params['ref_pa'] = pa_data[params['objname']]
# else:
#     params['ref_pa'] = 1e4
# if params['objname'] in q_data:
#     params['ref_q'] = q_data[params['objname']]
# else:
#     params['ref_q'] = 1e4
# params['ref_pa'] = rp.get_prop(params['objname'],'Stellar PA (deg)')
# params['ref_q'] = rp.get_prop(params['objname'],'Stellar Flattening')
params['ref_pa'] = 500
params['ref_q'] = -1.0

######################
# Import moment data #
######################

# Import dataframe
moment_data = pd.read_csv(params['data_filename'], skiprows=12)
# Import metadata
moment_data_full = pd.read_csv(params['data_filename'], nrows=7)
freq_obs = moment_data_full.iloc[5,0]
freq_obs = float(freq_obs[freq_obs.find(":")+1:])
bmaj = moment_data_full.iloc[1,0]
bmaj = float(bmaj[bmaj.find(":")+1:])
bmin = moment_data_full.iloc[2,0]
bmin = float(bmin[bmin.find(":")+1:])
bpa = moment_data_full.iloc[3,0]
bpa = float(bpa[bpa.find(":")+1:])
pix_scale = moment_data_full.iloc[0,0]
pix_scale = float(pix_scale[pix_scale.find(":")+1:])

pix_to_arcsec = pix_scale * 3600
pix_to_parsec = pix_to_arcsec * params['distance']*u.Mpc.to(u.pc) * np.pi / 180 / 3600

beam_area_pix = 1.1331 * bmaj * bmin / pix_scale**2
beam_area_arcsec = 1.1331 * bmaj * bmin * 3600**2

# Find flux center if necessary
if params['center_method'] != 'fixed':
    params['x0'], params['y0'] = func.centroid(moment_data)
## Do it anyway for SB plotting later
fx0, fy0 = func.centroid(moment_data)

# Remove bad bins
## From flux threshold
below_cutoff = moment_data['mom0 (Jy/pix km/s)'] <= params['flux_cutoff']
moment_data = moment_data[~below_cutoff]
## From bad_bins keyword
if type(params['bad_bins']) is int:
    moment_data.drop(int(params['bad_bins'])-14, inplace=True)
else:
    moment_data.drop([x-13 for x in list(params['bad_bins'])], inplace=True)
## From S/N threshold
below_cutoff = moment_data['S/N'] < params['snr_cutoff']
moment_data = moment_data[~below_cutoff]

xbin = moment_data['x (pix)'].values
ybin = moment_data['y (pix)'].values
velbin = moment_data['mom1 (km/s)'].values
fluxbin = moment_data['mom0 (Jy/pix km/s)'].values
velbin_unc = moment_data['mom1unc (km/s)'].values
fluxbin_unc = moment_data['mom0unc (Jy/pix km/s)'].values

binsize = moment_data['bin size (pix)'].values

####################################
# Find total brightness + gas mass #
####################################

# Find total intensity and uncertainty
intensity = sum( [fb*bs for fb,bs in zip(fluxbin,binsize)] )
intensity_stat_unc = np.sqrt(sum( [(fbu*bs)**2 for fbu,bs in zip(fluxbin_unc,binsize)] ))
intensity_unc = np.sqrt( intensity_stat_unc**2 + (0.1*intensity)**2 )  # Add with absolute uncertainty

if params['calc_mass'] and (params['redshift'] is not None) and (params['distance'] is not None):

    # Find CO luminosity (Boizelle et al. 2017 eq. 1)
    distance = params['distance']
    distance_unc = params['distance_unc'] if params['distance_unc'] is not None else 0.0
    freq_obs = freq_obs*u.Hz.to(u.GHz)
    lum_trans = 3.25e7 * intensity * (distance**2) / ((1+params['redshift'])**3*freq_obs**2)
    lum_trans_unc = lum_trans * np.sqrt( ( intensity_unc / intensity )**2 + ( 2*distance_unc/distance )**2 )

    # Convert to CO(1-0) luminosity
    ## Get user input for line
    rco = 0.0
    print(' ')
    print('Here\'s a list of transitions:')
    print(' ')
    for key in transitions:
        print('\t'+key+" : "+transitions[key])
    print(' ')
    trans_select = input('Select a transition with the corresponding letter (for gas mass estimate) : ')
    print(' ')
    ## Convert this to an R value
    if trans_select.lower() == "a":
        rco = 1.0
    elif trans_select.lower() == "b":
        rco = 0.7
    elif trans_select.lower() == "c":
        rco = 0.49
    ## Convert luminosity
    lum_co10 = lum_trans / rco
    lum_co10_unc = lum_trans_unc / rco

    # Find H2 mass
    alphaco = 3.1
    mass_H2 = alphaco * lum_co10
    mass_H2_unc = alphaco * lum_co10_unc

    # Find total gas mass
    f_He = 0.36
    mass_gas = mass_H2 * (1+f_He)
    mass_gas_unc = mass_H2_unc * (1+f_He)

    # Encapsulate this in one multiplicative factor
    intensity_to_mass = (1+f_He) * alphaco / rco * 3.25e7 * distance**2 / ((1+params['redshift'])**3*freq_obs**2)

#####################################
# Interpolate between bin centroids #
#####################################

# Make output grid
outx, outy = np.meshgrid(
        np.arange(int(np.min(xbin))-params['extrap_pixels'],int(np.max(xbin))+params['extrap_pixels']), 
        np.arange(int(np.min(ybin))-params['extrap_pixels'],int(np.max(ybin))+params['extrap_pixels'])
        )
outxy = np.column_stack([outx.ravel(), outy.ravel()])
interp_points = np.column_stack([xbin.ravel(),ybin.ravel()])

## Find the convex hull and make a mask for this
interp_hull = ConvexHull(interp_points)
hull_vert = interp_points[interp_hull.vertices]
outer_path = Path(hull_vert, closed=True)
inside_mask = outer_path.contains_points(outxy, radius=params['extrap_pixels'])
outxy = outxy[inside_mask]

# Interpolate
xyi = np.transpose(np.concatenate([[xbin],[ybin]]))
rbf_flux = interp.RBFInterpolator(xyi, fluxbin, kernel='thin_plate_spline')
rbf_vel = interp.RBFInterpolator(xyi, velbin, kernel='thin_plate_spline')
nn_velunc = interp.NearestNDInterpolator(xyi, velbin_unc)
nn_fluxunc = interp.NearestNDInterpolator(xyi, fluxbin_unc)
flux_interp = rbf_flux(outxy)
vel_interp = rbf_vel(outxy)
vel_unc_interp = nn_velunc(outxy)
flux_unc_interp = nn_fluxunc(outxy)

# Update values for kinemetry
xbin = outxy[:,0]
ybin = outxy[:,1]
fluxbin = flux_interp
velbin = vel_interp
velbin_unc = vel_unc_interp
fluxbin_unc = flux_unc_interp

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
        ring=params['ring']/pix_to_arcsec, verbose=params['verbose']
        )

#####################
# Monte Carlo it up #
#####################

pos_dist = {}
if params['mc']:
    num_runs_mc = 100
    pbmc = tqdm(range(num_runs_mc), desc="Running MC", leave=False)
    run_list = []
    velkin = k.velkin
    velkin_mask = velkin > 1e7
    velkin[velkin_mask] = velbin[velkin_mask]
    for itr in range(num_runs_mc):
        # Add random noise to original kinemetry model
        noised_model = np.random.normal(velkin, velbin_unc, np.size(xbin))
        # Rerun kinemetry
        run_list.append(kin.kinemetry(xbin=xbin, ybin=ybin, moment=noised_model, error=velbin_unc,
            x0=params['x0'], y0=params['y0'],
            rangeQ=params['rangeq'], rangePA=params['rangepa'],
            nq=params['nq'], npa=params['npa'],
            ntrm=params['ntrm'], incrad=params['incrad'],
            fixcen=params['fixcen'], nrad=params['nrad'],
            allterms=params['allterms'], even=params['even'],
            cover=params['cover'], plot=params['plot'],
            vsys=params['vsys'], drad=params['drad'],
            ring=params['ring']/pix_to_arcsec, verbose=False
            ))
        pbmc.update(1)
    pa_list = []
    q_list = []
    k1_list = []
    for run in run_list:
        pa_list.append(run.pa)
        q_list.append(run.q)
        k1_list.append(np.sqrt(run.cf[:,1]**2 + run.cf[:,2]**2))
    pa_list = np.array(pa_list)
    q_list = np.array(q_list)
    k1_list = np.array(k1_list)
    pa_std = np.std(pa_list, axis=0)
    q_std = np.std(q_list, axis=0)
    k1_std = np.std(k1_list, axis=0)
    pa_med = np.median(pa_list, axis=0)
    q_med = np.median(q_list, axis=0)
    k1_med = np.median(k1_list, axis=0)
    pos_dist = {
            "pa_med" : pa_med,
            "pa_std" : pa_std,
            "q_med" : q_med,
            "q_std" : q_std,
            "k1_med" : k1_med,
            "k1_std" : k1_std
            }

    # cp_labels = ['PA0', 'Q0', 'PA1', 'Q1', 'PA2', 'Q2', 'PA3', 'Q3']
    # cp_samples = np.transpose(np.array([
    #                                 pa_list[:,4],q_list[:,4],
    #                                 pa_list[:,5],q_list[:,5],
    #                                 pa_list[:,6],q_list[:,6],
    #                                 pa_list[:,7],q_list[:,7],
    #                             ]))
    # fig = corner.corner(
    #     cp_samples,
    #     labels=cp_labels,
    #     quantiles=[0.16,0.5,0.84],
    #     show_titles=True,
    #     plot_datapoints=True
    # )
    # plt.show()


#########################################################
# Get surface brightness profile with kinemetry results #
#########################################################

sbrad = np.logspace(np.log10(max(k.rad)/22), np.log10(max(k.rad)), num=20, base=10)
sbq = np.mean(k.q) * np.ones_like(sbrad)
sbpa = np.mean(k.pa) * np.ones_like(sbrad)

sb = kin.kinemetry(xbin=xbin, ybin=ybin, moment=fluxbin, error=fluxbin_unc,
        radius=sbrad, paq=np.array([np.mean(k.pa),np.mean(k.q)]),
        x0=fx0, y0=fy0, fixcen=True,
        plot=False, verbose=False)

##########################
# Plot and retrieve data #
##########################

# Plot radial profiles
radial_data = plotter.plot_kinemetry_profiles(k, pix_to_arcsec, pix_to_parsec, ref_pa=params['ref_pa'], ref_q=params['ref_q'], beam_size=beam_area_arcsec, pos_dist=pos_dist,
        user_plot_lims={
            "pa":params["plotlimspa"],
            "q":params["plotlimsq"],
            "k1":params["plotlimsk1"],
            "k5k1":params["plotlimsk5k1"]}
        )
if (params['saveloc'] != 'none') and params['saveplots']:
    plt.savefig(params['saveloc']+params['objname']+'_'+params['linename']+'_radial_profiles.png', dpi=1000)

# Plot surface brightness profile
sb_data = plotter.plot_sb_profiles(sb, intensity_to_mass, beam_area_pix, pix_to_arcsec, pix_to_parsec)
if (params['saveloc'] != 'none') and params['saveplots']:
    plt.savefig(params['saveloc']+params['objname']+'_'+params['linename']+'_sb_profile.png', dpi=1000)

# Plot v_los maps
spatial_data = plotter.plot_vlos_maps(xbin, ybin, velbin, k)
if (params['saveloc'] != 'none') and params['saveplots']:
    plt.savefig(params['saveloc']+params['objname']+'_'+params['linename']+'_velocity_maps.png', dpi=1000)
    
# Plot summary sheet
summary = plotter.plot_summary(k, pix_to_arcsec, pix_to_parsec, params['dataloc'], ref_pa=params['ref_pa'], ref_q=params['ref_q'], beam_size=beam_area_arcsec, bmin=bmin/pix_scale, bmaj=bmaj/pix_scale, bpa=bpa,
        user_plot_lims={
            "pa":params["plotlimspa"],
            "q":params["plotlimsq"],
            "k1":params["plotlimsk1"],
            "k5k1":params["plotlimsk5k1"]}
        )
if (params['saveloc'] != 'none') and params['saveplots']:
    plt.savefig(params['saveloc']+params['objname']+'_'+params['linename']+'_summary.png', dpi=1000)

# If not saving figures, just show
if not params['saveloc']:
    plt.show()
if params['saveloc'] and not (params['saveplots'] or params['savedata']):
    logging.warning('Save location set but nothing to save! Did you mean to set \'saveplots\' or \'savedata\'?')

##############
# Write data #
##############

# Save plot data
if params['savedata']:
    radial_data.to_csv(params['saveloc']+params['objname']+'_'+params['linename']+'_radial_data.csv', index=True)
    spatial_data.to_csv(params['saveloc']+params['objname']+'_'+params['linename']+'_spatial_data.csv', index=False)

    # Save csv of kinemetry parameters
    k1 = list(radial_data['k1'])
    dk1 = list(radial_data['dk1'])
    pa = list(radial_data['pa'])
    dpa = list(radial_data['dpa'])
    q = list(radial_data['q'])
    dq = list(radial_data['dq'])
    k5k1 = list(radial_data['k5k1'])
    dk5k1 = list(radial_data['dk5k1'])
    kin_params = pd.DataFrame(
            {
                'range k1 (km/s)' : [max(k1)-min(k1)],
                'range k1 uncertainty (km/s)' : [ np.sqrt( dk1[np.argmax(k1)]**2 + dk1[np.argmin(k1)]**2 ) ],
                'max k1 (km/s)' : [max(k1)],
                'max k1 uncertainty (km/s)' : [dk1[np.argmax(k1)]],
                'average k5k1' : [np.mean(k5k1)],
                'average k5k1 uncertainty' : [ np.sqrt(sum( [ kk**2 for kk in k5k1 ] )) ],
                'average pa (deg)' : [np.mean(pa)],
                'average pa uncertainty (deg)' : [ np.sqrt(sum( [ p**2 for p in pa ] )) ],
                'range pa (deg)' : [max(pa)-min(pa)],
                'range pa uncertainty (deg)' : [ np.sqrt( dpa[np.argmax(pa)]**2 + dpa[np.argmin(pa)]**2 ) ],
                'average q' : [np.mean(q)],
                'average q uncertainty' : [ np.sqrt(sum( [ qq**2 for qq in q ] )) ],
                'range q' : [max(q)-min(q)],
                'range q uncertainty' : [ np.sqrt( dq[np.argmax(q)]**2 + dq[np.argmin(q)]**2 ) ],
                'inclination (deg)' : [min(q)],
                'inclination uncertainty (deg)' : [dq[np.argmin(min(q))]],
                'intensity (Jy km/s)' : [intensity],
                'intensity uncertainty (Jy km/s)' : [intensity_unc],
                'luminosity (K km/s pc^2)' : [lum_trans],
                'luminosity uncertainty (K km/s pc^2)' : [lum_trans_unc],
                'gas mass (M_sol)' : [mass_gas],
                'gas mass uncertainty (M_sol)' : [mass_gas_unc]
            }
        )
    kin_params.to_csv(params['saveloc']+params['objname']+'_'+params['linename']+'_kinemetry_parameters.csv', index=False)

    # Save also to a json with all the kinemetry parameters
    alldata = {}
    try:
        alldata = json.load(open(params['saveloc']+'all_parameters.json'))
    except FileNotFoundError:
        print(' ')
        print("kinemetry parameters file not found . . . creating a new one here")
    alldata[params['objname']] = {
                'range k1 (km/s)' : max(k1)-min(k1),
                'range k1 uncertainty (km/s)' : np.sqrt( dk1[np.argmax(k1)]**2 + dk1[np.argmin(k1)]**2 ),
                'max k1 (km/s)' : max(k1),
                'max k1 uncertainty (km/s)' : dk1[np.argmax(k1)],
                'average k5k1' : np.mean(k5k1),
                'average k5k1 uncertainty' : np.sqrt(sum( [ kk**2 for kk in k5k1 ] )),
                'average pa (deg)' : np.mean(pa),
                'average pa uncertainty (deg)' : np.sqrt(sum( [ p**2 for p in pa ] )),
                'range pa (deg)' : max(pa)-min(pa),
                'range pa uncertainty (deg)' : np.sqrt( dpa[np.argmax(pa)]**2 + dpa[np.argmin(pa)]**2 ),
                'average q' : np.mean(q),
                'average q uncertainty' : np.sqrt(sum( [ qq**2 for qq in q ] )),
                'range q' : max(q)-min(q),
                'range q uncertainty' : np.sqrt( dq[np.argmax(q)]**2 + dq[np.argmin(q)]**2 ),
                'inclination (deg)' : min(q),
                'inclination uncertainty (deg)' : dq[np.argmin(min(q))],
                'intensity (Jy km/s)' : intensity,
                'intensity uncertainty (Jy km/s)' : intensity_unc,
                'luminosity (K km/s pc^2)' : lum_trans,
                'luminosity uncertainty (K km/s pc^2)' : lum_trans_unc,
                'gas mass (M_sol)' : mass_gas,
                'gas mass uncertainty (M_sol)' : mass_gas_unc
            }
    kp_outfile = open(params['saveloc']+'all_parameters.json', 'w')
    json.dump(alldata, kp_outfile, indent=5)
