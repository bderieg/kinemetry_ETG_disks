######################################################################################################
# Some of these functions are adapted from the 'run_kinemetry_examples.py' script by Davor Krajnovic #
######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from plotbin.plot_velfield import plot_velfield

from matplotlib.patches import Ellipse

import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

def plot_kinemetry_profiles_velocity(k, fitcentre=False, name=None):
    """
    Based on the kinemetry results (passed in k), this routine plots radial
    profiles of the position angle (PA), flattening (Q), k1 and k5 terms.
    Last two plots are for X0,Y0 and systemic velocity
    
    """

    k0 = k.cf[:,0]
    k1 = np.sqrt(k.cf[:,1]**2 + k.cf[:,2]**2)
    k5 = np.sqrt(k.cf[:,5]**2 + k.cf[:,6]**2)
    k51 = k5/k1
    erk1 = (np.sqrt( (k.cf[:,1]*k.er_cf[:,1])**2 + (k.cf[:,2]*k.er_cf[:,2])**2 ))/k1
    erk5 = (np.sqrt( (k.cf[:,5]*k.er_cf[:,5])**2 + (k.cf[:,6]*k.er_cf[:,6])**2 ))/k5
    erk51 = ( np.sqrt( ((k5/k1) * erk1)**2 + erk5**2  ) )/k1


    fig,ax =plt.subplots(figsize=(7,8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1,1,1])


    ax1 = plt.subplot(gs[0])
    ax1.errorbar(k.rad, k.pa, yerr=[k.er_pa, k.er_pa], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax1.set_ylabel('PA [deg]', fontweight='bold')
    if name:
        ax1.set_title(name, fontweight='bold')

    ax1.tick_params(axis='both', which='both', top=True, right=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_tick_params(length=6)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    ax1.xaxis.set_tick_params(length=6)
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='minor', width=1)
    ax1.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)
    ax1.get_xaxis().set_ticklabels([])

    ax2 = plt.subplot(gs[1])
    ax2.errorbar(k.rad, k.q, yerr=[k.er_q, k.er_q], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax2.set_ylabel('Q ', fontweight='bold')
    #ax2.set_xlabel('R [arsces]')
    ax2.set_ylim(0,1)
    if fitcentre:
        ax2.set_title('Velocity, fit centre', fontweight='bold')
    else:
        ax2.set_title('Velocity, fixed centre', fontweight='bold')


    ax2.tick_params(axis='both', which='both', top=True, right=True)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.yaxis.set_tick_params(length=6)
    ax2.xaxis.set_tick_params(width=2)
    ax2.yaxis.set_tick_params(width=2)
    ax2.xaxis.set_tick_params(length=6)
    ax2.tick_params(which='minor', length=3)
    ax2.tick_params(which='minor', width=1)
    ax2.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax2.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2)
    ax2.get_xaxis().set_ticklabels([])



    ax3 = plt.subplot(gs[2])
    ax3.errorbar(k.rad, k1, yerr=[erk1, erk1], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax3.set_ylabel('$k_1$ [km/s]', fontweight='bold')

    ax3.tick_params(axis='both', which='both', top=True, right=True)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.yaxis.set_tick_params(length=6)
    ax3.xaxis.set_tick_params(width=2)
    ax3.yaxis.set_tick_params(width=2)
    ax3.xaxis.set_tick_params(length=6)
    ax3.tick_params(which='minor', length=3)
    ax3.tick_params(which='minor', width=1)
    ax3.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax3.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(2)
    ax3.get_xaxis().set_ticklabels([])


    ax4 = plt.subplot(gs[3])
    ax4.errorbar(k.rad, k51, yerr=[erk51, erk51], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax4.set_ylabel('$k_{51}$', fontweight='bold')
    ax4.set_xlabel('R [arsces]', fontweight='bold')

    ax4.tick_params(axis='both', which='both', top=True, right=True)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    ax4.yaxis.set_tick_params(length=6)
    ax4.xaxis.set_tick_params(width=2)
    ax4.yaxis.set_tick_params(width=2)
    ax4.xaxis.set_tick_params(length=6)
    ax4.tick_params(which='minor', length=3)
    ax4.tick_params(which='minor', width=1)
    ax4.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax4.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax4.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax4.spines[axis].set_linewidth(2)
    ax4.get_xaxis().set_ticklabels([])

    ax5 = plt.subplot(gs[4])
    ax5.errorbar(k.rad, k.xc, yerr=k.er_xc, fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3, label='Xc')
    ax5.errorbar(k.rad, k.yc, yerr=k.er_yc, fmt='--o', mec='k', mew=1.2, color='salmon', mfc='salmon', capsize=3, label='Yc')
    ax5.set_ylabel('$X_c, Y_c$ [arsces]', fontweight='bold')
    ax5.set_xlabel('R [arsces]', fontweight='bold')
    ax5.legend()

    ax5.tick_params(axis='both', which='both', top=True, right=True)
    ax5.tick_params(axis='both', which='major', labelsize=10)
    ax5.yaxis.set_tick_params(length=6)
    ax5.xaxis.set_tick_params(width=2)
    ax5.yaxis.set_tick_params(width=2)
    ax5.xaxis.set_tick_params(length=6)
    ax5.tick_params(which='minor', length=3)
    ax5.tick_params(which='minor', width=1)
    ax5.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax5.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax5.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax5.spines[axis].set_linewidth(2)


    ax6 = plt.subplot(gs[5])
    ax6.errorbar(k.rad, k0, yerr=k.er_cf[:,0], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax6.hlines(np.median(k0),5,20,linestyles='dashed', colors='skyblue', label='median $V_{sys}$')
    ax6.set_ylabel('V$_{sys}$ [km/s]', fontweight='bold')
    ax6.set_xlabel('R [arsces]', fontweight='bold')
    ax6.legend()

    ax6.tick_params(axis='both', which='both', top=True, right=True)
    ax6.tick_params(axis='both', which='major', labelsize=10)
    ax6.yaxis.set_tick_params(length=6)
    ax6.xaxis.set_tick_params(width=2)
    ax6.yaxis.set_tick_params(width=2)
    ax6.xaxis.set_tick_params(length=6)
    ax6.tick_params(which='minor', length=3)
    ax6.tick_params(which='minor', width=1)
    ax6.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax6.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax6.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax6.spines[axis].set_linewidth(2)

    fig.tight_layout()
    return fig

def plot_kinemetry_profiles(k):
    # Retrieve relevant kinemetry outputs
    radii = k.rad
    pa = k.pa
    er_pa = k.er_pa
    q = k.pa
    er_q = k.er_q
    k1 = np.sqrt(k.cf[:,1]**2 + k.cf[:,2]**2)
    er_k1 = 1 - (np.sqrt(k.cf[:,1]**2 + k.cf[:,2]**2))/np.sqrt((k.cf[:,1]+k.er_cf[:,1])**2 + (k.cf[:,2]+k.er_cf[:,2])**2)
    k5 = np.sqrt(k.cf[:,5]**2 + k.cf[:,6]**2)
    er_k5 = 1 - (np.sqrt(k.cf[:,5]**2 + k.cf[:,6]**2))/np.sqrt((k.cf[:,5]+k.er_cf[:,5])**2 + (k.cf[:,6]+k.er_cf[:,6])**2)
    k5k1 = k5/k1
    er_k5k1 = 1 - (np.sqrt(k.cf[:,1]**2+k.cf[:,2]**2)*np.sqrt((k.cf[:,5]+k.er_cf[:,5])**2+(k.cf[:,6]+k.er_cf[:,6])**2))/(np.sqrt((k.cf[:,1]+k.er_cf[:,1])**2+(k.cf[:,2]+k.er_cf[:,2])**2)*np.sqrt(k.cf[:,5]**2+k.cf[:,6]**2))

    # Set up figure architecture
    fig = plt.figure()
    gs = fig.add_gridspec(4, hspace=0)
    ax = gs.subplots(sharex=True)

    # Plot pa
    ax[0].errorbar(radii, pa, yerr=er_pa, fmt='.k')
    ax[0].set_ylabel('$\Gamma$ (deg)')
    ax[0].set_box_aspect(1)

    # Plot q
    ax[1].errorbar(radii, q, yerr=er_q, fmt='.k')
    ax[1].set_ylabel('$q$')
    ax[1].set_box_aspect(1)

    # Plot k1
    ax[2].errorbar(radii, k1, yerr=list(map(lambda x,y:x*y,er_k1,k1)), fmt='.k')
    ax[2].set_ylabel('$k_1$ (km s$^{-1}$)')
    ax[2].set_box_aspect(1)

    # Plot k5k1
    ax[3].errorbar(radii, k5k1, yerr=list(map(lambda x,y:x*y,er_k5k1,k5k1)), fmt='.k')
    ax[3].set_xlabel('Radius (arcsec)')
    ax[3].set_ylabel('$k_5/k_1$')
    ax[3].set_box_aspect(1)

    # Set title
    fig.suptitle('NGC 2974 Kinemetry')
    
    return fig

def plot_vlos_maps(xbin, ybin, velbin, k, sigma=False):
    # Get some values for plotting
    k0 = k.cf[:,0]
    k1 = np.sqrt(k.cf[:,1]**2 + k.cf[:,2]**2)
    vsys = np.median(k0)
    mx = np.max(k1)
    mn = -mx
    if sigma:
        mx = np.max(k0)
        mn = np.min(k0)
        vsys = 0

    # Describe a mask for unfit pixels
    mask = np.where(k.velcirc < 12345679)

    # Set up figure architecture
    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0)

    # Plot observed moment
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('Data', x=-0.7, y=0.5, fontdict={"fontsize":15})
    ax1.set_aspect(1)
    ax1.axis('off')
    plot_velfield(xbin[mask], ybin[mask], velbin[mask]-vsys, colorbar=False, nodots=True, vmin=mn, vmax=mx)

    # Plot first-order fit
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    ax2.set_title('Kinemetry', x=-0.7, y=0.5, fontdict={"fontsize":15})
    ax2.set_aspect(1)
    ax2.axis('off')
    plot_velfield(xbin[mask], ybin[mask], k.velcirc[mask]-vsys, colorbar=False, nodots=True, vmin=mn, vmax=mx)

    # Plot residuals
    ax3 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax1)
    ax3.set_title('Residuals', x=-0.7, y=0.5, fontdict={"fontsize":15})
    ax3.set_aspect(1)
    ax3.axis('off')
    plot_velfield(xbin[mask], ybin[mask], list(map(lambda x,y:x-y,k.velcirc[mask],velbin[mask])), colorbar=False, nodots=True, vmin=mn, vmax=mx)

    # Set title
    fig.suptitle('NGC 2974 $v_{LOS}$', fontsize=15)

    return fig

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
    ax2.set_title('Kinemetry', x=-0.7, y=0.5, fontdict={"fontsize":15})
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
    fig.suptitle('NGC 2974 Surface Brightness', fontsize=15)

    return fig
