######################################################################################################
# Some of these functions are adapted from the 'run_kinemetry_examples.py' script by Davor Krajnovic #
######################################################################################################

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import ticker
from plotbin.plot_velfield import plot_velfield
import pandas as pd

from matplotlib.patches import Ellipse

import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['xtick.labelsize'] = 'small'
mpl.rcParams['ytick.labelsize'] = 'small'
mpl.rcParams['axes.labelpad'] = 10

##############################################################
# Function to plot radial profiles based on kinemetry output #
#   Parameters:                                              #
#       k: the output kinemetry object                       #
#   Returns:                                                 #
#       data = pandas dataframe with nicely organized        #
#           kinemetry outputs                                #
##############################################################

def plot_kinemetry_profiles(k, scale, phys_scale, ref_pa=None, ref_q=None, beam_size=None, user_plot_lims={}):

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
    ax[0].errorbar(radii, pa, yerr=dpa, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7, zorder=1)
    ax[0].fill_between(radii, k.pa_md-k.pa_sp, k.pa_md+k.pa_sp, fc='lightgray', zorder=0)
    ax[0].set_ylabel('$\Gamma$ (deg)', rotation='horizontal', ha='right')
    ax[0].set_box_aspect(0.5)
    ax[0].set_xlim(left=0)
    ax[0].yaxis.tick_right()
    ax[0].set_ylim(plot_lims["pa"])
    ax[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[0].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ## Plot reference pa if applicable
    if ref_pa is not None:
        ax[0].axhline(y=ref_pa, color='red', ls='dashed')

    ## Plot physical scale on top
    axphys = ax[0].twiny()
    axphys.set_box_aspect(0.5)
    axphys.set_xlabel('Radius (pc)')
    axphys.set_xlim([i/scale*phys_scale for i in ax[0].get_xlim()])
    axphys.xaxis.set_major_locator(ticker.MaxNLocator(4))

    # Plot q
    ax[1].errorbar(radii, q, yerr=dq, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7, zorder=1)
    ax[1].fill_between(radii, k.q_md-k.q_sp, k.q_md+k.q_sp, fc='lightgray', zorder=0)
    ax[1].set_ylabel('$q$', rotation='horizontal', ha='left')
    ax[1].set_box_aspect(0.5)
    ax[1].yaxis.set_label_position('right')
    ax[1].set_ylim(plot_lims["q"])
    ax[1].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[1].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ## Plot reference q if applicable
    if ref_q is not None:
        ax[1].axhline(y=ref_q, color='red', ls='dashed')

    # Plot k1
    ax[2].errorbar(radii, k1, yerr=dk1, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7, zorder=1)
    ax[2].fill_between(radii, k.k1_md-k.k1_sp, k.k1_md+k.k1_sp, fc='lightgray', zorder=0)
    ax[2].set_ylabel('$k_1$ (km s$^{-1}$)', rotation='horizontal', ha='right')
    ax[2].set_box_aspect(0.5)
    ax[2].yaxis.tick_right()
    ax[2].set_ylim(plot_lims["k1"])
    ax[2].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[2].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ## Plot beam reference line if applicable
    if beam_size is not None:
        if np.arccos(min(q))*(180/np.pi) < 75:
            ax[2].axvline(x=2*beam_size, color='blue', ls='dotted')
        else:
            ax[2].axvline(x=5*beam_size, color='blue', ls='dotted')

    # Plot k5k1
    ax[3].errorbar(radii, k5k1, yerr=dk5k1, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7)
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
