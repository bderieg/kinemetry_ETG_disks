######################################################################################################
# Some of these functions are adapted from the 'run_kinemetry_examples.py' script by Davor Krajnovic #
######################################################################################################

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import ticker
from plotbin.plot_velfield import plot_velfield
import pandas as pd
from astropy.io import fits
import scipy.interpolate as spint

from matplotlib.patches import Ellipse

import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['xtick.labelsize'] = 7.
mpl.rcParams['ytick.labelsize'] = 7.
mpl.rcParams['axes.labelsize'] = 8.
mpl.rcParams['axes.labelpad'] = 7.
mpl.rcParams['axes.titlesize'] = 8.

##############################################################
# Function to plot radial profiles based on kinemetry output #
#   Parameters:                                              #
#       k: the output kinemetry object                       #
#   Returns:                                                 #
#       data = pandas dataframe with nicely organized        #
#           kinemetry outputs                                #
##############################################################

def plot_kinemetry_profiles(k, scale, phys_scale, m_bh=0.0, model_data={}, ref_pa=None, ref_q=None, beam_size=None, user_plot_lims={}, pos_dist={}):

    # Retrieve kinemetry outputs and calculate uncertainties
    radii = k.rad[:]*scale

    ## v_sys (i.e., k_0)
    k0 = k.cf[:,0]
    dk0 = k.cf[:,0]

    ## position angle
    pa = k.pa[:]
    dpa = k.er_pa[:]

    ## flattening
    q = k.q[:]
    dq = k.er_q[:]

    ## k1 (i.e., sqrt(A_1^2 + B_1^2))
    k1 = np.sqrt(k.cf[:,1]**2 + k.cf[:,2]**2)
    ### intermediate uncertainty propagation calculations
    da1 = 2 * k.er_cf[:,1] * np.abs(k.cf[:,1])
    db1 = 2 * k.er_cf[:,2] * np.abs(k.cf[:,2])
    dk1_sq = np.sqrt(da1**2 + db1**2)
    ### final uncertainty calculation
    dk1 = 0.5 * dk1_sq * np.abs(k1**-1)

    ## k5 (i.e., sqrt(A_5^2 + B_5^2))
    k5 = np.sqrt(k.cf[:,5]**2 + k.cf[:,6]**2)
    ### intermediate uncertainty propagation calculations
    da5 = 2 * k.er_cf[:,5] * np.abs(k.cf[:,5])
    db5 = 2 * k.er_cf[:,6] * np.abs(k.cf[:,6])
    dk5_sq = np.sqrt(da5**2 + db5**2)
    ### final uncertainty calculation
    dk5 = 0.5 * dk5_sq * np.abs(k5**-1)

    ## k5/k1 (and uncertainty propagation)
    k5k1 = k5/k1
    dk5k1 = np.abs(k5k1) * np.sqrt((dk1/k1)**2 + (dk5/k5)**2)

    # Set up figure architecture
    fig = plt.figure()
    gs = fig.add_gridspec(4, hspace=0)
    ax = gs.subplots(sharex=True)
    markersize = 2

    # Set default plot limits
    plot_lims = {
            "pa" : [min(pa)-0.1*(max(pa)-min(pa)), max(pa)+0.1*(max(pa)-min(pa))],
            "q" : [min(q)-0.1*(max(q)-min(q)), max(q)+0.1*(max(q)-min(q))],
            "k1" : [min(k1)-0.1*(max(k1)-min(k1)), max(k1)+0.1*(max(k1)-min(k1))],
            "k5k1" : [0.00, 0.04]
            }
    if 'rad' in model_data:
        plot_lims = {
                "pa" : [
                        min(min(pa)-0.1*(max(pa)-min(pa)), min(model_data['pa'])-0.1*(max(model_data['pa'])-min(model_data['pa']))), 
                        max(max(pa)+0.1*(max(pa)-min(pa)), max(model_data['pa'])+0.1*(max(model_data['pa'])-min(model_data['pa'])))
                    ],
                "q" : [
                        min(min(q)-0.1*(max(q)-min(q)), min(model_data['q'])-0.1*(max(model_data['q'])-min(model_data['q']))),
                        max(max(q)+0.1*(max(q)-min(q)), max(model_data['q'])+0.1*(max(model_data['q'])-min(model_data['q'])))
                    ],
                "k1" : [
                        min(min(k1)-0.1*(max(k1)-min(k1)), min(model_data['v_ext'])-0.1*(max(model_data['v_ext'])-min(model_data['v_ext']))),
                        max(max(k1)+0.3*(max(k1)-min(k1)), max(model_data['v_ext'])+0.3*(max(model_data['v_ext'])-min(model_data['v_ext'])))
                    ],
                "k5k1" : [0.00, 0.04]
                }
    ## Override with user limits
    ### First remove None values from user dictionary
    user_plot_lims = {key : value for key, value in user_plot_lims.items() if value is not None}
    plot_lims |= user_plot_lims

    # Plot pa
    ax[0].errorbar(radii, pa, yerr=dpa, marker='s', c='black', ms=markersize, lw=0.0, elinewidth=0.7, zorder=1)
    if 'rad' in model_data:
        ax[0].errorbar(model_data['rad'], model_data['pa'], yerr=model_data['pa_unc'], fmt='k', marker='s', mec='black', mfc='white', ms=markersize, lw=0.0, elinewidth=0.7, zorder=1)
    if "pa_med" in pos_dist:
        ax[0].fill_between(radii, pos_dist["pa_med"]-pos_dist["pa_std"], pos_dist["pa_med"]+pos_dist["pa_std"], ec=None, fc='lightgray', zorder=0)
    ax[0].set_ylabel('$\Gamma$ (deg)', rotation='horizontal', ha='right')
    ax[0].set_box_aspect(0.5)
    ax[0].set_xlim(left=0)
    ax[0].yaxis.tick_right()
    ax[0].set_ylim(plot_lims["pa"])
    ax[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[0].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ## Plot physical scale on top
    axphys = ax[0].twiny()
    axphys.set_box_aspect(0.5)
    axphys.set_xlabel('Radius (pc)')
    axphys.set_xlim([i/scale*phys_scale for i in ax[0].get_xlim()])
    axphys.xaxis.set_major_locator(ticker.MaxNLocator(4))

    # Plot q
    ax[1].errorbar(radii, q, yerr=dq, marker='s', c='black', ms=markersize, lw=0.0, elinewidth=0.7, zorder=1, label='kinemetry')
    if 'rad' in model_data:
        ax[1].errorbar(model_data['rad'], model_data['q'], yerr=model_data['q_unc'], fmt='k', marker='s', mec='black', mfc='white', ms=markersize, lw=0.0, elinewidth=0.7, zorder=1, label='model')
    if "q_med" in pos_dist:
        ax[1].fill_between(radii, pos_dist["q_med"]-pos_dist["q_std"], pos_dist["q_med"]+pos_dist["q_std"], ec=None, fc='lightgray', zorder=0)
    ax[1].set_ylabel('$q$', rotation='horizontal', ha='left')
    ax[1].set_box_aspect(0.5)
    ax[1].yaxis.set_label_position('right')
    ax[1].set_ylim(plot_lims["q"])
    ax[1].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[1].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    if 'rad' in model_data:
        ax[1].legend(loc='upper left', bbox_to_anchor=(1.17,1.0), fontsize=6)

    # Plot k1
    ax[2].errorbar(radii, k1, yerr=dk1, marker='s', c='black', ms=markersize, lw=0.0, elinewidth=0.7, zorder=1)
    if 'rad' in model_data:
        ax[2].errorbar(
                    model_data['rad'], 
                    model_data['v_ext']*np.sqrt(1-model_data['q']**2), 
                    yerr=np.sqrt(model_data['v_ext_unc']*(1-model_data['q']**2)+(model_data['v_ext']**2*np.abs(model_data['q']*model_data['q_unc'])**2)/(1-model_data['q']**2)), 
                    fmt='k', marker='s', mec='black', mfc='white', ms=markersize, ls='-', lw=0.7, elinewidth=0.7, zorder=1
                )
    if "k1_med" in pos_dist:
        ax[2].fill_between(radii, pos_dist["k1_med"]-pos_dist["k1_std"], pos_dist["k1_med"]+pos_dist["k1_std"], ec=None, fc='lightgray', zorder=0)
    ax[2].set_ylabel('$k_1$ (km s$^{-1}$)', rotation='horizontal', ha='right')
    ax[2].set_box_aspect(0.5)
    ax[2].yaxis.tick_right()
    ax[2].set_ylim(plot_lims["k1"])
    ax[2].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[2].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    if 'rad' in model_data:
        ## Interpolate v_ext,q values
        q_interp = spint.interp1d(model_data['rad'], model_data['q'], fill_value='extrapolate')
        vext_interp = spint.interp1d(model_data['rad'], model_data['v_ext'], fill_value='extrapolate')
        ## Get BH curve
        G = 4.3009e-3
        finerad = np.arange(0.0, 10.0, 0.01)
        v_bh = np.sqrt(G*m_bh/(finerad/scale*phys_scale))
        ## Plot everything
        ax[2].plot(finerad, v_bh*np.sqrt(1-q_interp(finerad)**2), ls=':', c='black', label='$v_{BH}$', lw=0.7)
        ax[2].plot(finerad, vext_interp(finerad)*np.sqrt(1-q_interp(finerad)**2), ls='--', lw=0.7, c='black', zorder=-2, label='$v_{ext}$')
        ax[2].plot(finerad, np.sqrt(G*m_bh/(finerad/scale*phys_scale)+vext_interp(finerad)**2)*np.sqrt(1-q_interp(finerad)**2), ls='-', lw=0.7, c='black', zorder=-3, label='$v_{BH+ext}$')
        ax[2].legend(loc='upper left', bbox_to_anchor=(1.17,1.0), fontsize=6)

    # Plot k5k1
    ax[3].errorbar(radii, k5k1, yerr=dk5k1, marker='s', c='black', ms=markersize, lw=0.0, elinewidth=0.7, zorder=1)
    if "k5k1_med" in pos_dist:
        ax[3].fill_between(radii, pos_dist["k5k1_med"]-pos_dist["k5k1_std"], pos_dist["k5k1_med"]+pos_dist["k5k1_std"], ec=None, fc='lightgray', zorder=0)
    ax[3].set_xlabel('Radius (arcsec)')
    ax[3].set_ylabel('$k_5/k_1$', rotation='horizontal', ha='left')
    ax[3].set_box_aspect(0.5)
    ax[3].yaxis.set_label_position('right')
    ax[3].set_ylim(plot_lims["k5k1"])
    ax[3].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[3].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    fig.tight_layout()

    # Fill data and return
    data = pd.DataFrame(
                {
                    'pa' : pa,
                    'dpa' : dpa,
                    'q' : q,
                    'dq' : dq,
                    'k1' : k1,
                    'dk1' : dk1,
                    'k5k1' : k5k1,
                    'dk5k1' : dk5k1,
                    'k0' : k0,
                    'dk0' : dk0
                }
            )
    data.index = radii
    data.index.name = 'radius'
    return data


def plot_summary(k, ksb, ksb_lin, scale, phys_scale, dataloc, 
            ref_pa=None, ref_pa_unc=None, ref_q=None, beam_size=None, beam_area_pix=None,
            bmin=0, bmaj=0, bpa=0, intensity_to_mass=0, targetsn=0.0,
            user_plot_lims={}
        ):

    # Retrieve kinemetry outputs and calculate uncertainties
    radii = k.rad[:]*scale
    sbradii = ksb.rad[:]*scale
    linsbradii = ksb_lin.rad[:]*scale

    ## v_sys (i.e., k_0)
    k0 = k.cf[:,0]
    dk0 = k.cf[:,0]

    ## position angle
    pa = k.pa[:]
    dpa = k.er_pa[:]

    ## flattening
    q = k.q[:]
    dq = k.er_q[:]

    ## k1 (i.e., sqrt(A_1^2 + B_1^2))
    k1 = np.sqrt(k.cf[:,1]**2 + k.cf[:,2]**2)
    ### intermediate uncertainty propagation calculations
    da1 = 2 * k.er_cf[:,1] * np.abs(k.cf[:,1])
    db1 = 2 * k.er_cf[:,2] * np.abs(k.cf[:,2])
    dk1_sq = np.sqrt(da1**2 + db1**2)
    ### final uncertainty calculation
    dk1 = 0.5 * dk1_sq * np.abs(k1**-1)

    ## k5 (i.e., sqrt(A_5^2 + B_5^2))
    k5 = np.sqrt(k.cf[:,5]**2 + k.cf[:,6]**2)
    ### intermediate uncertainty propagation calculations
    da5 = 2 * k.er_cf[:,5] * np.abs(k.cf[:,5])
    db5 = 2 * k.er_cf[:,6] * np.abs(k.cf[:,6])
    dk5_sq = np.sqrt(da5**2 + db5**2)
    ### final uncertainty calculation
    dk5 = 0.5 * dk5_sq * np.abs(k5**-1)

    ## k5/k1 (and uncertainty propagation)
    k5k1 = k5/k1
    dk5k1 = np.abs(k5k1) * np.sqrt((dk1/k1)**2 + (dk5/k5)**2)

    ## surface brightness
    sb = ksb.cf[:,0] * beam_area_pix
    dsb = ksb.er_cf[:,0] * beam_area_pix
    linsb = ksb_lin.cf[:,0] * beam_area_pix
    lindsb = ksb_lin.er_cf[:,0] * beam_area_pix

    # Set up figure architecture
    fig = plt.figure(figsize=(8,4))
    subfigs = fig.subfigures(1, 2, wspace=-0.05, width_ratios=[1.,1.8])
    gs0 = subfigs[0].add_gridspec(4, hspace=0)
    rad_ax = gs0.subplots(sharex=True)
    radial_ratio = 0.7
    subfigs_right = subfigs[1].subfigures(3, 1, hspace=0.0, height_ratios=[0.8,1.,0.5])
    gsmaps = subfigs_right[0].add_gridspec(1, 3, wspace=0)
    maps_ax = gsmaps.subplots(sharey=True)
    histmaps = subfigs_right[1].add_gridspec(1, 3, wspace=0.8, bottom=0.3)
    hist_ax = histmaps.subplots()
    textgs = subfigs_right[2].add_gridspec(1)
    text_ax = textgs.subplots()

    # Set default plot limits
    plot_lims = {
            "pa" : [min(pa)-0.1*(max(pa)-min(pa)), max(pa)+0.1*(max(pa)-min(pa))],
            "q" : [min(q)-0.1*(max(q)-min(q)), max(q)+0.1*(max(q)-min(q))],
            "k1" : [min(k1)-0.1*(max(k1)-min(k1)), max(k1)+0.1*(max(k1)-min(k1))],
            "k5k1" : [0.00, 0.04]
            }
    ## Override with user limits
    ### First remove None values from user dictionary
    user_plot_lims = {key : value for key, value in user_plot_lims.items() if value is not None}
    plot_lims |= user_plot_lims

    # Plot pa
    rad_ax[0].errorbar(radii, pa, yerr=dpa, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7, zorder=1)
    rad_ax[0].set_ylabel('$\Gamma$ (deg)', rotation='horizontal', ha='right')
    rad_ax[0].set_box_aspect(radial_ratio)
    rad_ax[0].set_xlim(left=0)
    rad_ax[0].yaxis.tick_right()
    rad_ax[0].set_ylim(plot_lims["pa"])
    rad_ax[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    rad_ax[0].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ## Plot reference pa if applicable
    if ref_pa is not None:
        rad_ax[0].axhline(y=ref_pa, color='red', ls='dashed', zorder=0)
        rad_ax[0].fill_between(np.array([-1e9,1e9]), ref_pa-ref_pa_unc, ref_pa+ref_pa_unc, zorder=-1, fc=(1,0,0,0.1))

    ## Plot physical scale on top
    axphys = rad_ax[0].twiny()
    axphys.set_box_aspect(radial_ratio)
    axphys.set_xlabel('Radius (pc)')
    axphys.set_xlim([i/scale*phys_scale for i in rad_ax[0].get_xlim()])
    axphys.xaxis.set_major_locator(ticker.MaxNLocator(4))

    # Plot q
    rad_ax[1].errorbar(radii, q, yerr=dq, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7, zorder=1)
    rad_ax[1].set_ylabel('$q$', rotation='horizontal', ha='left')
    rad_ax[1].set_box_aspect(radial_ratio)
    rad_ax[1].yaxis.set_label_position('right')
    rad_ax[1].set_ylim(plot_lims["q"])
    rad_ax[1].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    rad_ax[1].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ## Plot reference q if applicable
    if ref_q is not None:
        rad_ax[1].axhline(y=ref_q, color='red', ls='dashed')

    # Plot k1
    rad_ax[2].errorbar(radii, k1, yerr=dk1, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7, zorder=1)
    rad_ax[2].set_ylabel('$k_1$ (km s$^{-1}$)', rotation='horizontal', ha='right')
    rad_ax[2].set_box_aspect(radial_ratio)
    rad_ax[2].yaxis.tick_right()
    rad_ax[2].set_ylim(plot_lims["k1"])
    rad_ax[2].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    rad_ax[2].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ## Plot beam reference line if applicable
    if beam_size is not None:
        if np.arccos(min(q))*(180/np.pi) < 75:
            rad_ax[2].axvline(x=2*beam_size, color='blue', ls='dotted')
        else:
            rad_ax[2].axvline(x=5*beam_size, color='blue', ls='dotted')

    # Plot k5k1
    rad_ax[3].errorbar(radii, k5k1, yerr=dk5k1, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7)
    rad_ax[3].set_xlabel('Radius (arcsec)')
    rad_ax[3].set_ylabel('$k_5/k_1$', rotation='horizontal', ha='left')
    rad_ax[3].yaxis.set_label_position('right')
    rad_ax[3].set_box_aspect(radial_ratio)
    rad_ax[3].set_ylim(plot_lims["k5k1"])
    rad_ax[3].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    rad_ax[3].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    # Plot moment maps
    ## Import maps
    mom0map = np.flipud(fits.open(dataloc+'mom0_binned.fits')[0].data)
    mom1map = np.flipud(fits.open(dataloc+'mom1_binned.fits')[0].data)
    mom2map = np.flipud(fits.open(dataloc+'mom2_binned.fits')[0].data)

    ## Import colormaps
    cmap0 = plt.cm.get_cmap('bone')
    cmap0.set_under('white')
    cmap1 = plt.cm.get_cmap('sauron')
    cmap1.set_under('white')

    ## Show moment 0
    mom0plot = maps_ax[0].imshow(
                mom0map, cmap=cmap0, 
                vmin=np.min(mom0map[np.nonzero(mom0map)]), 
                vmax=np.max(mom0map)
            )
    maps_ax[0].text(0.95*len(mom0map), 0.98*len(mom0map[0]), 'Mom 0', va='bottom', ha='right', fontsize=8.)
    maps_ax[0].axhline(y=0.95*len(mom0map), xmin=0.05, xmax=0.05+(1/scale)/len(mom0map[0]), color='black')
    ## Show beam reference
    beamEll = Ellipse((bmaj+1,bmaj+1), bmin, bmaj, angle=bpa, fc='gray', ec=None)
    maps_ax[0].add_patch(beamEll)
    ### Show colorbar
    cbar0 = subfigs_right[0].colorbar(mom0plot, ax=maps_ax[0], location='right', pad=0)
    cbar0.ax.set_yticks([])
    cbar0.ax.set_yticklabels([])
    ### Show value range
    maps_ax[0].text(
                0.95*len(mom0map), 
                0.03*len(mom0map), 
                str(round(1e3*np.min(mom0map[np.nonzero(mom0map)]),1))+' / '+str(round(1e3*np.max(mom0map),1))+' mJy', 
                va='top', ha='right', fontsize=6
            )

    ## Show moment 1
    mom1plot = maps_ax[1].imshow(
                mom1map, cmap=cmap1, 
                vmin=np.min(mom1map[np.nonzero(mom1map)]), 
                vmax=np.max(mom1map)
            )
    maps_ax[1].text(0.95*len(mom0map), 0.98*len(mom0map[0]), 'Mom 1', va='bottom', ha='right', fontsize=8.)
    ### Add compass
    compass_size = 0.1  # Adjust the size of the compass relative to the plot
    compass_x = 0.15*len(mom0map)
    compass_y = 0.95*len(mom0map[0])
    compass_end_x = 0.08*len(mom0map)
    compass_end_y = 0.88*len(mom0map)
    label_padding = 0.03*len(mom0map)
    maps_ax[1].plot([compass_x, compass_end_x], [compass_y, compass_y], c='black', lw=0.5)
    maps_ax[1].plot([compass_x, compass_x], [compass_y, compass_end_y], c='black', lw=0.5)
    label_padding = compass_size * maps_ax[1].get_xlim()[1] * 0.05
    maps_ax[1].text(compass_x, compass_end_y+label_padding, 'N', va='bottom', ha='center', fontsize=6.)
    maps_ax[1].text(compass_end_x-label_padding, compass_y, 'E', va='center', ha='right', fontsize=6.)
    ### Show PVD slice angle
    pvdangle = (fits.open(dataloc+'pvd.fits')[0].header)['CROTA2']
    if pvdangle > 180 : pvdangle -= 180
    pvdslope = -np.tan(pvdangle*(np.pi/180))
    maps_ax[1].axline((np.mean(k.xc),np.mean(k.yc)), slope=pvdslope, ls='--', c='black', lw=0.5)
    ### Show colorbar
    cbar1 = subfigs_right[0].colorbar(mom1plot, ax=maps_ax[1], location='right', pad=0)
    cbar1.ax.set_yticks([])
    cbar1.ax.set_yticklabels([])
    ### Show value range
    maps_ax[1].text(
                0.95*len(mom0map), 
                0.03*len(mom0map), 
                str(int(np.min(mom1map[np.nonzero(mom1map)])-np.mean(k0)))+' / '+str(int(np.max(mom1map)-np.mean(k0)))+' km s$^{-1}$', 
                va='top', ha='right', fontsize=6
            )

    ## Show moment 2
    mom2plot = maps_ax[2].imshow(
                mom2map, cmap=cmap0, 
                vmin=np.min(mom2map[np.nonzero(mom2map)]), 
                vmax=np.max(mom2map)
            )
    maps_ax[2].text(0.95*len(mom0map), 0.98*len(mom0map[0]), 'Mom 2', va='bottom', ha='right', fontsize=8.)
    ### Show colorbar
    cbar2 = subfigs_right[0].colorbar(mom2plot, ax=maps_ax[2], location='right', pad=0)
    cbar2.ax.set_yticks([])
    cbar2.ax.set_yticklabels([])
    ### Show value range
    maps_ax[2].text(
                0.95*len(mom0map), 
                0.03*len(mom0map), 
                str(round(np.min(mom2map[np.nonzero(mom2map)]),1))+' / '+str(round(np.max(mom2map),1))+' km s$^{-1}$', 
                va='top', ha='right', fontsize=6
            )

    for aa in maps_ax:
        aa.get_xaxis().set_visible(False)
        aa.get_yaxis().set_visible(False)

    # Plot velocity histogram
    veldata = pd.read_csv(dataloc+'vel_profile.csv')
    hist_ax[0].step(veldata.iloc[:,0].values, veldata.iloc[:,1].values, c='black', lw=0.5)
    hist_ax[0].set_box_aspect(1)
    hist_ax[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    hist_ax[0].set_xlabel('Velocity Channel\n(km s$^{-1}$)')
    hist_ax[0].set_ylabel('Flux Density (mJy)')

    # Plot PVD
    pvdmap = np.flipud(fits.open(dataloc+'pvd.fits')[0].data)
    pvdhdr = fits.open(dataloc+'pvd.fits')[0].header
    posref = pvdhdr['CRPIX1']
    delpos = pvdhdr['CRDELT1']
    velref = pvdhdr['CRPIX2']
    delvel = pvdhdr['CRDELT2']
    hist_ax[1].xaxis.set_major_formatter(lambda x,pos : round((x-posref)*delpos*3600))
    hist_ax[1].yaxis.set_major_formatter(lambda x,pos : -int((x-velref)*delvel))
    hist_ax[1].xaxis.set_major_locator(ticker.MaxNLocator(3))
    hist_ax[1].yaxis.set_major_locator(ticker.MaxNLocator(3))
    hist_ax[1].imshow(np.arcsinh(pvdmap), cmap='bone_r', aspect='auto')
    hist_ax[1].set_box_aspect(1)
    hist_ax[1].set_xlabel('Major Axis Distance\n(arcsec)')
    hist_ax[1].set_ylabel('$\Delta$ Velocity(km s$^{-1}$)')

    # Plot SB profile
    hist_ax[2].errorbar(sbradii, sb, yerr=dsb, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7)
    hist_ax[2].errorbar(linsbradii, linsb, yerr=lindsb, fmt='.r', markersize=1.5, linewidth=1, elinewidth=0.7)
    hist_ax[2].set_xscale('log')
    hist_ax[2].set_yscale('log')
    hist_ax[2].set_ylabel('$\log_{10}$ I/A$_\mathrm{beam}$\n(Jy km s$^{-1}$ beam$^{-1}$)')
    hist_ax[2].set_box_aspect(1.0)
    hist_ax[2].set_xlabel('$\log_{10}$ R\n(arcsec)')
    hist_ax[2].autoscale()
    hist_ax[2].xaxis.set_major_locator(ticker.LogLocator(base=10))
    hist_ax[2].xaxis.set_minor_formatter(lambda x,pos : None)
    hist_ax[2].xaxis.set_major_formatter(lambda x,pos : int(np.log10(x)))
    hist_ax[2].yaxis.set_major_locator(ticker.LogLocator(base=10))
    hist_ax[2].yaxis.set_minor_formatter(lambda y,pos : None)
    hist_ax[2].yaxis.set_major_formatter(lambda y,pos : int(np.log10(y)))
    ## Plot gas surface density scale on right
    axdensity = hist_ax[2].twinx()
    axdensity.set_box_aspect(1.0)
    axdensity.set_ylabel('$\log_{10}$ $\Sigma\'_{\\mathrm{gas}}$ (M$_\\odot$ pc$^{-2}$)')
    axdensity.set_yscale('log')
    axdensity.autoscale()
    axdensity.yaxis.set_major_locator(ticker.LogLocator(base=10))
    axdensity.yaxis.set_minor_formatter(lambda y,pos : None)
    axdensity.yaxis.set_major_formatter(lambda y,pos : int(np.log10(y)))
    ## Plot physical scale on top
    axphys_gas = hist_ax[2].twiny()
    axphys_gas.set_box_aspect(1.0)
    axphys_gas.set_xlabel('$\log_{10}$ R (pc)')
    axphys_gas.set_xscale('log')
    axphys_gas.set_xlim([i/scale*phys_scale for i in hist_ax[2].get_xlim()])
    axphys_gas.autoscale()
    axphys_gas.xaxis.set_major_locator(ticker.LogLocator(base=10))
    axphys_gas.xaxis.set_minor_formatter(lambda x,pos : None)
    axphys_gas.xaxis.set_major_formatter(lambda x,pos : int(np.log10(x)))

    # Text for other parameters
    text_ax.set_xticks([])
    text_ax.set_yticks([])
    text_ax.set_xticklabels([])
    text_ax.set_yticklabels([])
    text_ax.axis('off')

    text_ax.axhline(1.0, 0, 1, ls='--', c='black')
    text_ax.text(0, 0.8, 'Velocity Binning : '+str(round(delvel,1))+' km s$^{-1}$', va='center', ha='left', fontsize=8)
    text_ax.text(0, 0.5, 'Self-Calibration   : ', va='center', ha='left', fontsize=8)
    text_ax.text(0, 0.2, 'Target S/N            : '+str(targetsn), va='center', ha='left', fontsize=8)


##############################################################
# Same as plot_kinemetry_profiles, but for mom 0 run         #
##############################################################

def plot_sb_profiles(k, intensity_to_mass, beam_area_pix, scale, phys_scale):

    # Retrieve kinemetry outputs and calculate uncertainties
    radii = k.rad[:]*scale

    ## surface brightness
    sb = k.cf[:,0] * beam_area_pix
    dsb = k.er_cf[:,0] * beam_area_pix

    # Set up figure architecture
    fig = plt.figure()
    gs = fig.add_gridspec(1, hspace=0)
    ax = gs.subplots(sharex=True)

    # Plot sb
    ax.errorbar(radii, sb, yerr=dsb, fmt='sk', markersize=5, linewidth=1, elinewidth=1.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('$\log_{10}$ I/A$_\mathrm{beam}$ (Jy km s$^{-1}$ beam$^{-1}$)')
    ax.set_box_aspect(1.0)
    ax.set_xlabel('$\log_{10}$ R (arcsec)')
    ax.autoscale()
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.xaxis.set_minor_formatter(lambda x,pos : None)
    ax.xaxis.set_major_formatter(lambda x,pos : int(np.log10(x)))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.yaxis.set_minor_formatter(lambda y,pos : None)
    ax.yaxis.set_major_formatter(lambda y,pos : int(np.log10(y)))

    ## Plot gas surface density scale on right
    axdensity = ax.twinx()
    axdensity.set_box_aspect(1.0)
    axdensity.set_ylabel('$\log_{10}$ $\Sigma\'_{\\mathrm{gas}}$ (M$_\\odot$ pc$^{-2}$)')
    axdensity.set_yscale('log')
    axdensity.autoscale()
    axdensity.yaxis.set_major_locator(ticker.LogLocator(base=10))
    axdensity.yaxis.set_minor_formatter(lambda y,pos : None)
    axdensity.yaxis.set_major_formatter(lambda y,pos : int(np.log10(y)))

    ## Plot physical scale on top
    axphys = ax.twiny()
    axphys.set_box_aspect(1.0)
    axphys.set_xlabel('$\log_{10}$ R (pc)')
    axphys.set_xscale('log')
    axphys.set_xlim([i/scale*phys_scale for i in ax.get_xlim()])
    axphys.autoscale()
    axphys.xaxis.set_major_locator(ticker.LogLocator(base=10))
    axphys.xaxis.set_minor_formatter(lambda x,pos : None)
    axphys.xaxis.set_major_formatter(lambda x,pos : int(np.log10(x)))

    fig.tight_layout()

    # Fill data and return
    data = pd.DataFrame(
                {
                    'sb' : sb,
                    'dsb' : dsb
                }
            )
    data.index = radii
    data.index.name = 'radius'
    return data

#######################################################
# Function to plot maps for v_los (moment 1) -- plots #
# data, 1-term cosine fit, and  residuals             #
#   Parameters:                                       #
#       xbin, ybin, velbin: same as other functions   #
#       k: kinemetry output object                    #
#       value_mask: an array of 1s and 0s, where all  #
#           1s will be masked                         #
#   Returns:                                          #
#       data = pandas dataframe with nicely organized #
#           bin values                                #
#######################################################

def plot_vlos_maps(xbin, ybin, velbin, k):
    # Get some values for plotting
    k0 = k.cf[:,0]
    k1 = np.sqrt(k.cf[:,1]**2 + k.cf[:,2]**2)
    vsys = np.median(k0)
    mx = np.max(k1)
    mn = -mx

    # Describe a mask for unfit pixels
    model_mask = np.where(k.velcirc < 12345679)

    # Set up figure architecture
    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0)

    # Plot observed moment
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('Data', x=-0.7, y=0.5, fontdict={"fontsize":15})
    ax1.set_aspect(1)
    ax1.axis('off')
    plot_velfield(xbin, ybin, velbin-vsys, colorbar=False, nodots=True, vmin=mn, vmax=mx, zorder=1)

    # Plot first-order fit
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    ax2.set_title('1-term cos fit', x=-0.7, y=0.5, fontdict={"fontsize":15})
    ax2.set_aspect(1)
    ax2.axis('off')
    plot_velfield(xbin[model_mask], ybin[model_mask], k.velcirc[model_mask]-vsys, colorbar=False, nodots=True, vmin=mn, vmax=mx, zorder=1)

    # Plot residuals
    ax3 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax1)
    ax3.set_title('Residuals', x=-0.7, y=0.5, fontdict={"fontsize":15})
    ax3.set_aspect(1)
    ax3.axis('off')
    plot_velfield(xbin[model_mask], ybin[model_mask], list(map(lambda x,y:x-y,k.velcirc[model_mask],velbin[model_mask])), colorbar=False, nodots=True, vmin=mn, vmax=mx, zorder=1)

    # Set title
    fig.suptitle('$v_{LOS}$', fontsize=15)

    # Fill data and return
    data = pd.DataFrame(
                {
                    'xbin' : xbin,
                    'ybin' : ybin,
                    'velbin' : velbin,
                    'velcirc' : k.velcirc,
                    'residuals' : list(map(lambda x,y:x-y, k.velcirc, velbin))
                }
            )
    return data

####################################################################
# Function to plot maps for surface brightness (moment 0) -- plots #
# data, 1-term constant fit, and  residuals                        #
#   Parameters:                                                    #
#       xbin, ybin, velbin: same as other functions                #
#       k: kinemetry output object                                 #
####################################################################

def plot_sb_maps(img, k):
    # Get some results from fitting
    x1 = int(np.median(k.xc)-np.max(k.rad))
    y1 = int(np.median(k.yc)-np.max(k.rad))
    x2 = int(np.median(k.xc)+np.max(k.rad))
    y2 = int(np.median(k.yc)+np.max(k.rad))
    ext = [x1, x2, y1, y2]
    peak = img[int(round(np.median(k.xc))), int(round(np.median(k.yc)))]
    levels = peak * 10**(-0.4*np.arange(0, 5, 0.5)[::-1]) # 0.5 mag/arcsec^2 steps

    # Describe a mask for unfit pixels
    masked = np.where(k.velcirc < 12345679, img, -1)

    # Set up figure architecture
    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0)

    # Plot observed moment
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('Data', x=-0.7, y=0.5, fontdict={"fontsize":15})
    ax1.set_aspect(1)
    ax1.axis('off')
    ax1.imshow(np.log10(masked), 
            vmin=np.min(np.log10(levels)), vmax=np.max(np.log10(levels)),
            origin='lower', extent=ext,
            cmap="pink")

    # Plot first-order fit
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title('1-term sin fit', x=-0.7, y=0.5, fontdict={"fontsize":15})
    ax2.set_aspect(1)
    ax2.axis('off')
    ax2.imshow(np.log10(k.velcirc), 
            vmin=np.min(np.log10(levels)), vmax=np.max(np.log10(levels)),
            origin='lower', extent=ext,
            cmap="pink")

    # Plot residuals
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title('Residuals', x=-0.7, y=0.5, fontdict={"fontsize":15})
    ax3.set_aspect(1)
    ax3.axis('off')
    ax3.imshow(np.log10(img)-np.log10(k.velcirc), 
            vmin=np.min(np.log10(levels)), vmax=np.max(np.log10(levels)),
            origin='lower', extent=ext,
            cmap="pink")

    # Set title
    fig.suptitle('Surface Brightness', fontsize=15)
