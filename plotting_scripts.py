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

def plot_kinemetry_profiles(k, scale):
    # Retrieve relevant kinemetry outputs
    k0 = k.cf[:,0]
    er_k0 = k.er_cf[:,0]
    radii = k.rad[:]*scale
    pa = k.pa[:]
    er_pa = k.er_pa[:]
    q = k.q[:]
    er_q = k.er_q[:]
    k1 = np.sqrt(k.cf[:,1]**2 + k.cf[:,2]**2)
    er_k1 = 1 - (np.sqrt(k.cf[:,1]**2 + k.cf[:,2]**2))/np.sqrt((k.cf[:,1]+k.er_cf[:,1])**2 + (k.cf[:,2]+k.er_cf[:,2])**2)
    k5 = np.sqrt(k.cf[:,5]**2 + k.cf[:,6]**2)
    er_k5 = 1 - (np.sqrt(k.cf[:,5]**2 + k.cf[:,6]**2))/np.sqrt((k.cf[:,5]+k.er_cf[:,5])**2 + (k.cf[:,6]+k.er_cf[:,6])**2)
    k5k1 = k5/k1
    er_k5k1 = 1 - (np.sqrt(k.cf[:,1]**2+k.cf[:,2]**2)*np.sqrt((k.cf[:,5]+k.er_cf[:,5])**2+(k.cf[:,6]+k.er_cf[:,6])**2))/(np.sqrt((k.cf[:,1]+k.er_cf[:,1])**2+(k.cf[:,2]+k.er_cf[:,2])**2)*np.sqrt(k.cf[:,5]**2+k.cf[:,6]**2))

    # Set up figure architecture
    fig = plt.figure()
    gs = fig.add_gridspec(5, hspace=0)
    ax = gs.subplots(sharex=True)

    # Plot pa
    ax[0].errorbar(radii, pa, yerr=er_pa, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7)
    ax[0].set_ylabel('$\Gamma$ (deg)', rotation='horizontal', ha='right')
    ax[0].set_box_aspect(0.5)
    ax[0].set_xlim(left=0)
    ax[0].yaxis.tick_right()
    ax[0].set_ylim([min(pa)-0.1*(max(pa)-min(pa)), max(pa)+0.1*(max(pa)-min(pa))])
    ax[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[0].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    # Plot q
    ax[1].errorbar(radii, q, yerr=er_q, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7)
    ax[1].set_ylabel('$q$', rotation='horizontal', ha='left')
    ax[1].set_box_aspect(0.5)
    ax[1].yaxis.set_label_position('right')
    ax[1].set_ylim([min(q)-0.1*(max(q)-min(q)), max(q)+0.1*(max(q)-min(q))])
    ax[1].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[1].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    # Plot k1
    ax[2].errorbar(radii, k1, yerr=er_k1, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7)
    ax[2].set_ylabel('$k_1$ (km s$^{-1}$)', rotation='horizontal', ha='right')
    ax[2].set_box_aspect(0.5)
    ax[2].yaxis.tick_right()
    ax[2].set_ylim([min(k1)-0.1*(max(k1)-min(k1)), max(k1)+0.1*(max(k1)-min(k1))])
    ax[2].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[2].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    # Plot k5k1
    ax[3].errorbar(radii, k5k1, yerr=list(map(lambda x,y:x*y, er_k5k1, k5k1)), fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7)
    ax[3].set_xlabel('Radius (arcsec)')
    ax[3].set_ylabel('$k_5/k_1$', rotation='horizontal', ha='left')
    ax[3].set_box_aspect(0.5)
    ax[3].yaxis.set_label_position('right')
    ax[3].set_ylim([min(k5k1)-0.1*(max(k5k1)-min(k5k1)), max(k5k1)+0.1*(max(k5k1)-min(k5k1))])
    ax[3].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[3].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    # Plot v_sys 
    ax[4].errorbar(radii, k0, yerr=er_k0, fmt='.k', markersize=1.5, linewidth=1, elinewidth=0.7)
    ax[4].set_ylabel('$v_{sys}$ (km s$^{-1}$)', rotation='horizontal', ha='right')
    ax[4].set_box_aspect(0.5)
    ax[4].yaxis.tick_right()
    ax[4].set_ylim([min(k0)-0.1*(max(k0)-min(k0)), max(k0)+0.1*(max(k0)-min(k0))])
    ax[4].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax[4].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    fig.tight_layout()

    # Fill data and return
    data = pd.DataFrame(
                {
                    'pa' : pa,
                    'er_pa' : er_pa,
                    'q' : q,
                    'er_q' : er_q,
                    'k1' : k1,
                    'er_k1' : er_k1,
                    'k5k1' : k5k1,
                    'er_k5k1' : er_k5k1,
                    'k0' : k0,
                    'er_k0' : er_k0
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

def plot_vlos_maps(xbin, ybin, velbin, k, value_mask=None):
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
    if value_mask is not None:
        ax1.imshow(value_mask, alpha=value_mask, zorder=2, cmap='Greys', vmin=1, vmax=1e9)

    # Plot first-order fit
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    ax2.set_title('1-term cos fit', x=-0.7, y=0.5, fontdict={"fontsize":15})
    ax2.set_aspect(1)
    ax2.axis('off')
    plot_velfield(xbin[model_mask], ybin[model_mask], k.velcirc[model_mask]-vsys, colorbar=False, nodots=True, vmin=mn, vmax=mx, zorder=1)
    if value_mask is not None:
        ax2.imshow(value_mask, alpha=value_mask, zorder=2, cmap='Greys', vmin=1, vmax=1e9)

    # Plot residuals
    ax3 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax1)
    ax3.set_title('Residuals', x=-0.7, y=0.5, fontdict={"fontsize":15})
    ax3.set_aspect(1)
    ax3.axis('off')
    plot_velfield(xbin[model_mask], ybin[model_mask], list(map(lambda x,y:x-y,k.velcirc[model_mask],velbin[model_mask])), colorbar=False, nodots=True, vmin=mn, vmax=mx, zorder=1)
    if value_mask is not None:
        ax3.imshow(value_mask, alpha=value_mask, zorder=2, cmap='Greys', vmin=1, vmax=1e9)

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
